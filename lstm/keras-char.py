import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
from attention_lstm import attention_3d_block
import numpy as np
import random
import io
import os
import json

mode = "TRAIN"
path = "data/sosizraks2.txt"

LSTM_DIM = 768
DROPOUT_VAL = 0.05

network = "LSTM2"  # LSTM, 2LSTM, {CuDNN: LSTM2, LSTM3, LSTM3-DEEP, LSTM2-ATTENTION, LSTM2-ATTENTION-FELIX, LSTM3-ATTENTION }
BATCH_SIZE = 128   # 64 or less 50-48-32-16, 96 - already bad weights update with small data
VAL_SPLIT = 0.05   # 0.05

maxlen = 20  # 40 is ok+-   # set corresponding TIME_STEPS in attention_lstm.py
step = 1  # 1 is ok, 2, 3... reduces input data size

# text = io.open(path, encoding='utf-8').read()
text = io.open(path, encoding='utf-8').read().lower()
print('corpus length:', len(text))

# https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions/27522708
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

del sentences
del next_chars

# https://github.com/philipperemy/keras-attention-mechanism/issues/14
# no Flatten() needed at the end of attention block !!!
from keras.layers import Reshape, Lambda, dot, concatenate

# INPUT_DIM = 100  # ??? not used here ???
# ATTENTION_SIZE = 128  # last layer size - attention_vector
TIME_STEPS = maxlen  # 20 was by default?
SINGLE_ATTENTION_VECTOR = True  # if True, the attention vector is shared across the input_dimensions where the attention is applied
APPLY_ATTENTION_BEFORE_LSTM = False


def attention_3d_block_felixhao28(hidden_states):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(hidden_states.shape[2])
    hidden_states = Reshape((TIME_STEPS, hidden_size), name='hidden_states')(hidden_states)
    # _t stands for transpose
    # hidden_states_t.shape = (batch_size, hidden_size, time_steps)
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    # score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    #            score_first_part_t         dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot (batch_size, hidden_size, 1) => (batch_size, time_steps, 1)
    h_t = Lambda(lambda x: x[:, -1:, :], output_shape=(1, hidden_size), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 2], name='attention_score')
    score = Reshape((TIME_STEPS,), name='attention_score_flat')(score)
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps, 1) => (batch_size, hidden_size, 1)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    h_t = Reshape((hidden_size,), name='last_hidden_state_flat')(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


print('Build model...')
model = Sequential()
if network == "LSTM":
    # Model 1: LSTM X*3
    model.add(LSTM(LSTM_DIM, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(LSTM(LSTM_DIM, return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(LSTM(LSTM_DIM))
    model.add(Dropout(DROPOUT_VAL))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))
if network == "2LSTM":
    # Model 1: LSTM X*2
    model.add(LSTM(LSTM_DIM, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(LSTM(LSTM_DIM))
    model.add(Dropout(DROPOUT_VAL))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))
elif network == "LSTM2":
    # Model 1: CuDNNLSTM X*2
    model.add(CuDNNLSTM(LSTM_DIM, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(CuDNNLSTM(LSTM_DIM))
    model.add(Dropout(DROPOUT_VAL))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))
elif network == "LSTM3":
    # Model 1: CuDNNLSTM X*3
    model.add(CuDNNLSTM(LSTM_DIM, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(CuDNNLSTM(LSTM_DIM, return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(CuDNNLSTM(LSTM_DIM))
    model.add(Dropout(DROPOUT_VAL))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))
elif network == "LSTM3-DEEP":
    # Model 1: CuDNNLSTM X*3 + Dense
    model.add(CuDNNLSTM(LSTM_DIM, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(CuDNNLSTM(LSTM_DIM, return_sequences=True))
    model.add(Dropout(DROPOUT_VAL))
    model.add(CuDNNLSTM(LSTM_DIM))
    model.add(Dropout(DROPOUT_VAL))
    model.add(Dense(LSTM_DIM))
    model.add(Dropout(DROPOUT_VAL))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))
elif network == "LSTM2-ATTENTION":
    # Model 2 - CuDNNLSTM X*2 + attention block
    from keras.models import Model
    from keras.layers import Input, Flatten

    inputs = Input(shape=(maxlen, len(chars),))
    lstm_out = CuDNNLSTM(LSTM_DIM, return_sequences=True)(inputs)
    lstm_out = Dropout(DROPOUT_VAL)(lstm_out)
    lstm_out = CuDNNLSTM(LSTM_DIM, return_sequences=True)(lstm_out)
    lstm_out = Dropout(DROPOUT_VAL)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(len(chars), activation='softmax')(attention_mul)
    model = Model(input=[inputs], output=output)
elif network == "LSTM2-ATTENTION-FELIX":
    # Model 2 - CuDNNLSTM X*2 + attention block
    from keras.models import Model
    from keras.layers import Input

    inputs = Input(shape=(maxlen, len(chars),))
    lstm_out = CuDNNLSTM(LSTM_DIM, return_sequences=True)(inputs)
    lstm_out = Dropout(DROPOUT_VAL)(lstm_out)
    lstm_out = CuDNNLSTM(LSTM_DIM, return_sequences=True)(lstm_out)
    lstm_out = Dropout(DROPOUT_VAL)(lstm_out)
    attention_mul = attention_3d_block_felixhao28(lstm_out)
    output = Dense(len(chars), activation='softmax')(attention_mul)
    model = Model(input=[inputs], output=output)
elif network == "LSTM3-ATTENTION":
    # Model 2 - CuDNNLSTM X*3 + attention block
    # https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
    from keras.models import Model
    from keras.layers import Input, Flatten

    inputs = Input(shape=(maxlen, len(chars),))
    lstm = CuDNNLSTM(LSTM_DIM, return_sequences=True)(inputs)
    lstm = Dropout(DROPOUT_VAL)(lstm)
    lstm = CuDNNLSTM(LSTM_DIM, return_sequences=True)(lstm)
    lstm = Dropout(DROPOUT_VAL)(lstm)
    lstm = CuDNNLSTM(LSTM_DIM, return_sequences=True)(lstm)
    lstm = Dropout(DROPOUT_VAL)(lstm)
    lstm = attention_3d_block(lstm)
    lstm = Flatten()(lstm)
    lstm = Dense(len(chars), activation='softmax')(lstm)
    model = Model(inputs=[inputs], outputs=lstm)

# compile model
# optimizer = RMSprop(lr=0.002, clipnorm=5.0, clipvalue=0.5, decay=0.97)   # 0.001, no decay rate    # decay not changes as we does .fit() 1 time only?
optimizer = Adam(lr=0.001, clipnorm=5.0, clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
model.summary()


# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    # preds = np.log(preds.clip(min=0.00001)) / temperature    # fix "divide by zero encountered in log"
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def logger(q, s):
    f = open("words_log.txt", "a+")
    f.write(q)
    f.write(s)
    f.write("\n")
    f.close()


# keras.js export chars
with open('_chars_.json', 'w') as fp:
    json.dump(chars, fp)

if os.path.isfile("lstm_chars.h5"):
    # model = load_model("lstm_chars.h5")   # for just Keras it's Ok
    model.load_weights("lstm_chars.h5")  # for Keras.js continued models

# convert CuDNNLSTM -> LSTM - for keras.js v3
model.save("lstm_chars_new.h5")  # just for model conversion
# exit()

# cheat for keras.js v2
# if os.path.isfile("weights_array.npy"):
#    print("\nLoading pretrained weights...Ok\n")
#    model.set_weights(np.load("weights_array.npy"))

# from keras.callbacks import TensorBoard
# tb_callback = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True, write_images=True)

# model.summary()
for iteration in range(1, 401):
    print()
    # print('-' * 70)
    print("Iteration", iteration)

    if mode == "TRAIN":
        model.fit(x, y, batch_size=BATCH_SIZE, epochs=1, validation_split=VAL_SPLIT, callbacks=[])
        model.save("lstm_chars.h5")
        # cheat for keras.js v2
        # np.save("weights_array", model.get_weights())

    if iteration % 10 == 0:
        start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.75]:
            # print()
            # print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            # print('----- Generating with seed: "' + sentence + '"')
            # sys.stdout.write(generated)

            gen_out = ""
            gen_orig = generated

            chars_to_generate = 15000
            if mode == "TRAIN":
                chars_to_generate = 500

            for i in range(chars_to_generate):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.0

                # print(len(x_pred[0][0]))
                # exit()

                predictions = model.predict(x_pred, verbose=0)[0]
                next_index = sample(predictions, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                gen_out += next_char

                # sys.stdout.write(next_char)
                # sys.stdout.flush()

            logger("----------- ", str(iteration) + " --- " + str(diversity) + " -----------")
            logger(gen_orig, gen_out)

        if mode != "TRAIN":
            exit()
