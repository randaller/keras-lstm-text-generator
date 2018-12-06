import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM, TimeDistributed, Embedding
# http://forums.fast.ai/uploads/default/optimized/1X/43a48f3ba8b0ace15c574f3a20a31e8eaf1396ed_1_690x497.png
from keras.optimizers import Adam, RMSprop, Adadelta, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import io
import json

# WARNING! Keras char-rnn versions are all fucked up!!!
# https://groups.google.com/forum/#!topic/keras-users/g88CwkuWXVQ

path = "data/piknik.txt"

LSTM_DIM = 512
DROPOUT_VAL = 0.2

BATCH_SIZE = 64

# text = io.open(path, encoding='utf-8').read()
text = io.open(path, encoding='utf-8').read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 1
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

print('Build model...')
model = Sequential()

# yxtay way - https://github.com/yxtay/char-rnn-text-generation

# model.add(Embedding(len(chars), 32, batch_input_shape=(128, maxlen)))
# model.add(Dropout(DROPOUT_VAL))
# model.add(LSTM(LSTM_DIM, return_sequences=False, stateful=True))

# 1 layer only

# model.add(LSTM(LSTM_DIM, input_shape=(maxlen, len(chars)), return_sequences=False))

# 3-layers LSTM

model.add(CuDNNLSTM(LSTM_DIM, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(DROPOUT_VAL))
model.add(CuDNNLSTM(LSTM_DIM, return_sequences=True))
model.add(Dropout(DROPOUT_VAL))
model.add(CuDNNLSTM(LSTM_DIM))
model.add(Dropout(DROPOUT_VAL))

# model.add(Dense(LSTM_DIM))        # extra Dense
# model.add(Dropout(DROPOUT_VAL))   # extra Dense

model.add(Dense(len(chars)))
model.add(Activation("softmax"))

optimizer = Adam(lr=0.001, clipnorm=5.0, clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

# keras.js export chars
with open('_chars_.json', 'w') as fp:
    json.dump(chars, fp)

checkpoint = ModelCheckpoint(filepath="lstm_chars_epoch{epoch:05d}_loss{val_loss:.4f}.h5", verbose=1, save_weights_only=False, save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=3)

#from sklearn.utils import shuffle
#x, y = shuffle(x, y)

model.fit(x, y, batch_size=BATCH_SIZE, epochs=1000, callbacks=[checkpoint, early_stop], validation_split=0.05)
