BATCH=$1
TEST_BATCH=$2
VOCAB=$3
MLEN=$4
DATA=$5
FIXED_EPOCHS=$6
NUM_EPOCHS=$7
NHIDDEN=$8
PRETRAINED=""
LAST_EPOCH=$9
SAMPLE_TIME=3
LEARNING_RATE=0.01
FLAGS="floatX=float32,device=gpu1"
THEANO_FLAGS="${FLAGS}" python main.py --model LSTM   --data ${DATA} --batch_size ${BATCH} --vocab_size ${VOCAB} --max_len ${MLEN} --fixed_epochs ${FIXED_EPOCHS} --num_epochs ${NUM_EPOCHS} --num_hidden ${NHIDDEN} --test_batch ${TEST_BATCH} --learning_rate ${LEARNING_RATE} --sample_time ${SAMPLE_TIME}
