# -*- coding:utf-8 -*-
#! /usr/bin/env python

from lasagne import init
from lasagne import nonlinearities

class OutGate(object):
    def __init__(self, W_in=init.Normal(0.1), W_hid=init.Normal(0.1),
                 W_cell=init.Normal(0.1), W_to=init.Normal(0.1),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        self.W_to = W_to
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

class TimeGate(object):

    def __init__(self, W_t=init.Normal(0.1), W_x=init.Normal(0.1),
            b=init.Constant(0.),
            nonlinearity_inside=nonlinearities.tanh,
            nonlinearity_outside=nonlinearities.sigmoid):
        self.W_t = W_t
        self.W_x = W_x
        self.b = b
        self.nonlinearity_inside = nonlinearity_inside
        self.nonlinearity_outside = nonlinearity_outside
