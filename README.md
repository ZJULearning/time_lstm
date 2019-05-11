# Time-LSTM on RecSys


# Introduction

This project is the implementation of the paper "What to Do Next: Modeling User Behaviors by Time-LSTM".

### Abstract of The Paper

>Recently, Recurrent Neural Network (RNN) solutions for recommender systems(RS) are becoming increasingly popular. The insight is that, there exist some intrinstic patterns in the sequence of users' behaviors, and RNN has been proved to perform excellently tasks such as language modeling, RNN solutions usually only consider sequential order of objects from a sequence without the notion of interval. However, in RS, time intervals between users' behaviors are of significant importance in capturing the relations of users' behaviors and the traditional RNN architectures are not good ar modeling them. In this paper, we propose a new LSTM variant, i.e. Time-LSTM, to model users' sequential behaviors. Time-LSTM equips LSTM with time gates to model time intervals. These time gates are specifically designed, so that compared to the traditional RNN solutions, Time-LSTM better captures both of users' short-term and long-term interest, so as to improve the recommendation performance. Experimental results on two real-world datasets show the superiority of the recommendation method using Time-LSTM over the traditional methods.

## Model

Time-LSTM proposed in the paper has three different architectures.

* TLSTM1: lstm with time gate 1.
* TLSTM2: lstm with time gate 1 & 2.
* TLSTM3: lstm with time gate 1 & 2, and forget gate is removed.

![](http://d.pr/i/nVrp+)

****************

## Requirments

The code is tested in the following envirenment.  
theano=0.9.0  
lasagne=0.2.dev1  
pandas=0.18.1  
cudnn=5.1  
cuda=8.0  

```bash
pip install -v theano==0.9.0
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install -v pandas==0.18.1
```

## Data Preprocess

* [Last.FM](description&download: https://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html):   
* [CiteULike](description: http://www.citeulike.org/faq/data.adp; download: https://pan.baidu.com/s/1ej4C4UBiAsRJqWheI5LEgw):   

Download and put the original data file into _data/music_ or _data/citeulike_.  
For the _citeulike_ dataset, you need run _awk -F "|" '{print $1"|"$2"|"$3}' citeulike_dataset | uniq > citeulike-origin-filtered_ to get _citeulike-origin-filtered_ and filter users and items with few interactions.  
Run the python file in the preprocess folder (e.g. _lastfm.py_) to get three files: _user-item.lst, user-item-delta-time.lst_ and _user-item-accumulate-time.lst_.  
Use head/tail command (e.g. _head -800 user-item.lst > tr\_user-item.lst_ and _tail -192 user-item.lst > te\_user-item.lst_) to generate the following 6 files for each(in data/{data_source}/).  

* tr(te)\_user-item.lst
* tr(te)\_user-item-delta-time.lst
* tr(te)\_user-item-accumulate-time.lst

_tr\_*_ is traing data. _te\_*_ is testing data. The content is something like: 

>user1, item1 item2 item3 ...  
user2, item1 item2 item3 ...  

## Files

plstm.py: phased-lstm  
tlstm1.py: TLSTM1  
tlstm2.py: TLSTM2  
tlstm3.py: TLSTM3  
lstm.py: LSTM  
utils.py: load data, save module, and some other method...  
main.py: main process to train.  

train.sh: command samples.


******************

To train the model, run:

```bash
THEANO_FLAGS="${FLAGS}" python main.py --model TLSTM3   --data ${DATA} --batch_size ${BATCH} --vocab_size ${VOCAB} --max_len ${MLEN} --fixed_epochs ${FIXED_EPOCHS} --num_epochs ${NUM_EPOCHS} --num_hidden ${NHIDDEN} --test_batch ${TEST_BATCH} --learning_rate ${LEARNING_RATE} --sample_time ${SAMPLE_TIME}
```
or just run (the hyperparameters in _train.sh_ are not carefully tuned):

```bash
bash train.sh
```

Important arguments:

* --model: 
    * LSTM: Using LSTM model.
    * LSTM_T: Using LSTM model and use time interval as a feature.
    * [PLSTM](https://arxiv.org/abs/1610.09513): Using PLSTM model.
    * TLSTM1: Using TLSTM1 model.
    * TLSTM2: Using TLSTM2 model.
    * TLSTM3: Using TLSTM3 model.
    
* --data:
    * music: Using Last.FM as the data source.
    * citeulike: Using CiteULike as data source.
