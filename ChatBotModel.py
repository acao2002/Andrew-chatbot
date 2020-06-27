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
    "name" : 1,
    "age" : 2,
    "origin": 3,
    "greeting" : 4,
}
outputLength = len(dictlabel)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
word_index = tokenizer.word_index
total_words = len(word_index)
print(total_words)

training_sequences = tokenizer.texts_to_sequences(questions)
max_sequence_len = max([len(x) for x in training_sequences])
training_padded = pad_sequences(training_sequences, maxlen=max_sequence_len, padding="pre", truncating="post")


training_labels = []; 
for item in intentions: 
    training_labels.append(dictlabel[item])

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)

print(training_padded)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 16, input_length= max_sequence_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(training_padded,training_labels, epochs= 20)
