import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("AnQnA.json", 'r') as f:
    datastore = json.load(f)

    questions = []
    intentions = []

for item in datastore:
    questions.append(item['Question'].lower().replace("?", " "))
    intentions.append(item['Intention'])

dictlabel = {
    "name" : 0,
    "age" : 1,
    "origin": 2,
    "greeting" : 3,
    "hru" : 4,
    "school" : 5 ,
    "music" : 6, 
}
outputLength = len(dictlabel)
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', oov_token ="OOV")
tokenizer.fit_on_texts(questions)
word_index = tokenizer.word_index
total_words = len(word_index)


training_sequences = tokenizer.texts_to_sequences(questions)
max_sequence_len = max([len(x) for x in training_sequences])
training_padded = pad_sequences(training_sequences, maxlen=max_sequence_len, padding="pre", truncating="post")


training_labels = []; 
for item in intentions: 
    training_labels.append(dictlabel[item])

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
training_labels = tf.keras.utils.to_categorical(training_labels, num_classes= 7)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(9,)),
    tf.keras.layers.Embedding(total_words+1, 100, input_length= max_sequence_len, trainable = False),
    tf.keras.layers.LSTM(150),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax'),
    
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(training_padded,training_labels, epochs= 200)

model.save('ChatBot.h5')
print(total_words)
print(len(training_labels))