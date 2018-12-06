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

maxlen = 40

path = "data/ddt.txt"

LSTM_DIM = 128
LAYERS = 3
DROPOUT_VAL = 0.2
ADD_DROP = True
ADD_EXTRA_DENSE = False

BATCH_SIZE = 32

try:
    text = open(path).read().lower()
except UnicodeDecodeError:
    import codecs

    text = codecs.open(path, "r", encoding='utf-8').read().lower()
print('corpus length:', len(text))

##### old version, before reduce #####
# words = set(open(path, "r", encoding="utf-8").read().lower().split())
# VOCABULARY_SIZE = len(words) ##### old version, before reduce #####
# print("words", type(words))
# print("total number of unique words", len(words))

##### old version, before reduce #####
# word_indices = dict((c, i) for i, c in enumerate(words))
# indices_word = dict((i, c) for i, c in enumerate(words))
# print("word_indices", type(word_indices), "length:", len(word_indices))
# print("indices_words", type(indices_word), "length", len(indices_word))

list_words = text.lower().split()  # TODO: split better
digit_words = []
del (text)

##### old version, before reduce #####
# for i in range(0, len(list_words)):
#     digit_words.append(word_indices[list_words[i]])
# del (list_words)

### START REDUCE ###
# Reduce words list to only 20000 most frequent words
wordcount = {}
for i in range(0, len(list_words)):
    digit = list_words[i]
    if digit not in wordcount:
        wordcount[digit] = 1
    else:
        wordcount[digit] += 1

import collections

word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(10):
    print(word, ": ", count)

new_words = {"<UNK>": 0}
cnt = 1
for word, count in word_counter.most_common(19999):
    new_words[word] = cnt
    cnt += 1

# print(type(new_words))
# print(new_words)

VOCABULARY_SIZE = len(new_words)

word_indices = dict((c, i) for i, c in enumerate(new_words))
indices_word = dict((i, c) for i, c in enumerate(new_words))
print("word_indices", type(word_indices), "length:", len(word_indices))
print("indices_words", type(indices_word), "length", len(indices_word))

for i in range(0, len(list_words)):
    if list_words[i] in word_indices:
        digit_words.append(word_indices[list_words[i]])
    else:
        digit_words.append(0)
del (list_words)


# exit()
### END REDUCE ###

def describe_batch(X, y, samples=3):
    for i in range(samples):
        sentence = ""
        for s in range(maxlen):
            sentence += indices_word[X[i, s, :].argmax()] + " "
        next_char = indices_word[y[i, :].argmax()]

        print("sample #%d: ...%s -> '%s'" % (i, sentence[-30:], next_char))


def batch_generator(text, count):
    while True:
        for batch_ix in range(count):
            X = np.zeros((BATCH_SIZE, maxlen, VOCABULARY_SIZE))
            y = np.zeros((BATCH_SIZE, VOCABULARY_SIZE))

            batch_offset = BATCH_SIZE * batch_ix

            for sample_ix in range(BATCH_SIZE):
                sample_start = batch_offset + sample_ix

                for s in range(maxlen):
                    X[sample_ix, s, text[sample_start + s]] = 1

                y[sample_ix, text[sample_start + s + 1]] = 1

            yield X, y


print("Build model...")
model = Sequential()

model.add(LSTM(LSTM_DIM, return_sequences=True, input_shape=(maxlen, len(new_words))))
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

model.add(Dense(len(new_words)))
model.add(Activation("softmax"))

optimizer = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)
model.summary()

# keras.js export words as chars for website
with open("_chars_.json", "w") as fp:
    json.dump(list(new_words), fp)

checkpoint = ModelCheckpoint(filepath="lstm_words_epoch{epoch:04d}_loss{loss:.4f}.h5", verbose=1, save_weights_only=False, save_best_only=False)
early_stop = EarlyStopping(monitor="loss", patience=20)

text_train_len = len(digit_words)
train_batch_count = (text_train_len - maxlen) // BATCH_SIZE
print("training batch count: %d" % train_batch_count)

for ix, (X, y) in enumerate(batch_generator(digit_words, count=1)):
    describe_batch(X, y, samples=1)
    break

history = model.fit_generator(
    batch_generator(digit_words, count=train_batch_count),
    train_batch_count,
    max_queue_size=1,  # no more than one queued batch in RAM
    epochs=10000,
    callbacks=[checkpoint, early_stop],
    # validation_data=batch_generator(digit_words_val, count=val_batch_count),
    # validation_steps=val_batch_count,
    initial_epoch=0
)
