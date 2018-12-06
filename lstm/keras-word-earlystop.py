import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import numpy as np

path = "data/ddt.txt"

LSTM_DIM = 128
LAYERS = 2
DROPOUT_VAL = 0.2
ADD_DROP = False
ADD_EXTRA_DENSE = False

BATCH_SIZE = 64

try:
    text = open(path).read().lower()
except UnicodeDecodeError:
    import codecs

    text = codecs.open(path, "r", encoding='utf-8').read().lower()

print('corpus length:', len(text))

chars = set(text)
words = set(open(path, "r", encoding="utf-8").read().lower().split())

print("chars:", type(chars))
print("words", type(words))
print("total number of unique words", len(words))
print("total number of unique chars", len(chars))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

print("word_indices", type(word_indices), "length:", len(word_indices))
print("indices_words", type(indices_word), "length", len(indices_word))

maxlen = 10
step = 1
print("maxlen:", maxlen, "step:", step)

next_words = []
sentences = []
list_words = text.lower().split()

for i in range(0, len(list_words) - maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    sentences.append(sentences2)
    next_words.append((list_words[i + maxlen]))
print("nb sequences (length of sentences):", len(sentences))
print("length of next_word", len(next_words))

print("Vectorization...")
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        # print(i,t,word)
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

print("Build model...")
model = Sequential()

#from keras.layers.normalization import BatchNormalization
#model.add(BatchNormalization(input_shape=(maxlen, len(words))))

model.add(LSTM(LSTM_DIM, return_sequences=True, input_shape=(maxlen, len(words))))
if ADD_DROP:
    model.add(Dropout(DROPOUT_VAL))

if LAYERS == 3:
    model.add(LSTM(LSTM_DIM, return_sequences=True))
    if ADD_DROP:
        model.add(Dropout(DROPOUT_VAL))

model.add(LSTM(LSTM_DIM, return_sequences=False))
if ADD_DROP:
    model.add(Dropout(DROPOUT_VAL))

if ADD_EXTRA_DENSE:
    model.add(Dense(LSTM_DIM))
    model.add(Dropout(DROPOUT_VAL))

model.add(Dense(len(words)))
model.add(Activation("softmax"))

optimizer = Adam(lr=0.001, clipnorm=5.0, clipvalue=0.5)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['acc'])
model.summary()

# keras.js export words as chars for website
with open("_chars_.json", "w") as fp:
    json.dump(list(words), fp)

checkpoint = ModelCheckpoint(filepath="lstm_words_epoch{epoch:05d}_loss{loss:.4f}_val{val_loss:.4f}.h5", verbose=1, save_weights_only=False, save_best_only=False)
early_stop = EarlyStopping(monitor="loss", patience=20)

# from sklearn.utils import shuffle
# X, y = shuffle(X, y)

model.fit(X, y, batch_size=BATCH_SIZE, epochs=10000, callbacks=[checkpoint, early_stop], validation_split=0.1)
