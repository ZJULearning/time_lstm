# -*- coding: utf-8 -*-
#! /usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers.recurrent import Gate
from tgate import OutGate, TimeGate

class TLSTM3Layer(MergeLayer):

    def __init__(self, incoming,time_input, num_units,
                 ingate=Gate(),
                 tgate1=TimeGate(W_t=init.Uniform((-1,0))),
                 tgate2=TimeGate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=OutGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 bn=False,
                 boundary=-0.00001, # constraint ceil
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        incomings.append(time_input)
        self.time_incoming_index = len(incomings) - 1

        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(TLSTM3Layer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.boundary = boundary

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        time_shape = self.input_shapes[1]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        def add_outgate_params(gate, gate_name):
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.W_to, (1, num_units),
                                   name="W_to_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        def add_timegate_params(gate, gate_name):
            return (self.add_param(gate.W_t, (1, num_units), 
                        name="W_t_to_{}".format(gate_name)),
                    self.add_param(gate.W_x, (num_inputs, num_units),
                        name="W_x_to_{}".format(gate_name)),
                    self.add_param(gate.b,(num_units,),
                        name="b_{}".format(gate_name)),
                    gate.nonlinearity_inside,
                    gate.nonlinearity_outside
                    )


        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate,self.W_to_to_outgate,
         self.b_outgate,
         self.nonlinearity_outgate) = add_outgate_params(outgate, 'outgate')

        (self.W_t1_to_tg1, self.W_x1_to_tg1, self.b1_tg1, self.nonlinearity_inside_tg1,
         self.nonlinearity_outside_tg1) = add_timegate_params(tgate1, 'tgate1')

        (self.W_t2_to_tg2, self.W_x2_to_tg2, self.b2_tg2, self.nonlinearity_inside_tg2,
         self.nonlinearity_outside_tg2) = add_timegate_params(tgate2, 'tgate2')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        if bn:
            self.bn = lasagne.layers.BatchNormLayer(input_shape, axes=(0,1))
            self.params.update(self.bn.params)
        else:
            self.bn = False


    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # TLSTM: Define new input
        time_mat = inputs[self.time_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        if self.bn:
            input = self.bn.get_output_for(input)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        #(n_time_steps, n_batch)
        time_input = time_mat.dimshuffle(1, 0, 'x')
        time_seq_len, time_num_batch, _ = time_input.shape
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 5*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate,
             self.W_in_to_cell, self.W_in_to_outgate, self.W_x2_to_tg2, self.W_x1_to_tg1], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (5*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate,
             self.b_cell, self.b_outgate, self.b2_tg2, self.b1_tg1], axis=0)

        # W_t1_to_tg1_constraint < 0
        W_t1_to_tg1_constraint = T.switch(T.ge(self.W_t1_to_tg1, self.boundary), self.W_t1_to_tg1, self.boundary)

        # Stack delta time weight matrices into a (num_inputs, 2* num_units)
        W_t_stacked = T.concatenate([ self.W_to_to_outgate, self.W_t2_to_tg2, W_t1_to_tg1_constraint ], axis=1)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            time_input = T.dot(time_input, W_t_stacked)
            input = T.dot(input, W_in_stacked) + b_stacked

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, start, stride=1):
            return x[:, start*self.num_units:(start+stride)*self.num_units]


        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        # todo
        # insert Tm_n, weight_t_o_n in to mask_n and xell_previous
        def step(input_n, time_input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                time_input_n = T.dot(time_input_n, W_t_stacked)
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            tm_wto_n = slice_w(time_input_n, 0)
            tm_w2_n = slice_w(time_input_n, 1)
            tm_w1_n = slice_w(time_input_n, 2)
            tm_w2_n = self.nonlinearity_inside_tg2(tm_w2_n)
            tm_w1_n = self.nonlinearity_inside_tg1(tm_w1_n)
            tm2_xwb_n = slice_w(input_n, 3)
            tm1_xwb_n = slice_w(input_n, 4)
            timegate2 = self.nonlinearity_outside_tg2(tm_w2_n + tm2_xwb_n)
            timegate1 = self.nonlinearity_outside_tg1(tm_w1_n + tm1_xwb_n)
            input_n = slice_w(input_n, 0, 3)

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            cell_input = slice_w(gates, 1)
            outgate = slice_w(gates, 2)
            outgate += tm_wto_n

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            cell_input = self.nonlinearity_cell(cell_input)
            
            # Compute new cell value
            cell = (1 - ingate)*cell_previous + ingate*timegate2*cell_input
            tilde_cell = (1 - ingate*timegate1)*cell_previous + ingate*timegate1*cell_input

            if self.peepholes:
                outgate += tilde_cell*self.W_cell_to_outgate

            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(tilde_cell)
            return [cell, hid]

        def step_masked(input_n, time_input_n, mask_n, 
                cell_previous, hid_previous, *args):

            cell, hid = step(input_n, time_input_n, 
                    cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, time_input, mask] 
            step_fun = step_masked
        else:
            sequences = [input, time_input]
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]

        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked, W_t_stacked]
        else:
            pass

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out

