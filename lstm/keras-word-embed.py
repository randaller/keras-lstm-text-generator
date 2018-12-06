import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM, Embedding
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
import keras
import json
import numpy as np
import random
import sys
import os

mode = "TRAIN"
path = "data/aria.txt"

LSTM_DIM = 128
DROPOUT_VAL = 0.05

net = "LSTM-"
BATCH_SIZE = 96

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

maxlen = 40
step = 1
print("maxlen:", maxlen, "step:", step)

next_words = []
sentences = []
list_words = text.lower().split()

for i in range(0, len(list_words) - maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    sentences.append(sentences2)
    next_words.append((list_words[i + maxlen]))
print('nb sequences (length of sentences):', len(sentences))
print("length of next_word", len(next_words))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        # print(i,t,word)
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

if net == "LSTM":
    cell = LSTM
else:
    cell = CuDNNLSTM

print('Build model...')
model = Sequential()
model.add(Embedding(len(words), 50, input_length=maxlen))
model.add(cell(LSTM_DIM, return_sequences=True))
model.add(Dropout(DROPOUT_VAL))
# model.add(cell(LSTM_DIM, return_sequences=True))
# model.add(Dropout(DROPOUT_VAL))
model.add(cell(LSTM_DIM))
model.add(Dropout(DROPOUT_VAL))
# model.add(Dense(LSTM_DIM))
# model.add(Dropout(DROPOUT_VAL))
model.add(Dense(len(words)))
model.add(Activation("softmax"))

optimizer = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def logger(q, s):
    f = open('words_log.txt', 'a+')
    f.write(q)
    f.write(s)
    f.write("\n")
    f.close()


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    # preds = np.log(preds.clip(min=0.00001)) / temperature    # fix "divide by zero encountered in log"
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.loss = 0

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.loss = logs.get('loss')


history = LossHistory()

epoch_mul = 1
prev_loss = 100

# keras.js export words as chars for website
with open('_chars_.json', 'w') as fp:
    json.dump(list(words), fp)

if os.path.isfile("lstm_words.h5"):
    # model = load_model("lstm_words.h5")   # for just Keras it's Ok
    model.load_weights("lstm_words.h5")  # for Keras.js continued models - ( may be bad - models strange training after loading? )

# convert CuDNNLSTM -> LSTM - for keras.js v3
model.save("lstm_words_new.h5")  # just for model conversion
# exit()

# train the model, output generated text after each iteration
for iteration in range(1, 300):
    print()
    print('Iteration', iteration)
    if mode == "TRAIN":
        model.fit(X, y, batch_size=BATCH_SIZE, epochs=epoch_mul, callbacks=[history])
        model.save("lstm_words.h5", overwrite=True)

        logger("[ Epoch " + str(iteration * epoch_mul), ", Loss " + str(history.loss) + " ]")
        logger("", "")

    start_index = random.randint(0, len(list_words) - maxlen - 1)

    for diversity in [0.8]:
        print()
        print('----- diversity:', diversity)
        generated = ''
        sentence = list_words[start_index: start_index + maxlen]
        generated += ' '.join(sentence)
        print('----- Generating with seed: "', sentence, '"')
        print()
        sys.stdout.write(generated)
        print()

        words_to_generate = 2000
        if mode == "TRAIN":
            words_to_generate = 200

        output_text = ""
        for i in range(1, words_to_generate):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.0

            predictions = model.predict(x, verbose=0)[0]
            next_index = sample(predictions, diversity)
            next_word = indices_word[next_index]
            generated += next_word
            del sentence[0]
            sentence.append(next_word)
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()

            output_text += next_word
            output_text += " "
            if i % 20 == 0: output_text += "\n"

        logger("[ Temp " + str(diversity) + " ] ", output_text)
        logger("", "")
        print()

    if mode != "TRAIN":
        exit()

    # early stop (not for rmsprop, try for adam)
    '''
    if history.loss > prev_loss:
        print()
        print("early stop at", history.loss, prev_loss)
        break
    prev_loss = history.loss
    '''
