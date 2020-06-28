import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
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
}
outputLength = len(dictlabel)
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~' , oov_token= "OOV")
tokenizer.fit_on_texts(questions)
word_index = tokenizer.word_index
total_words = len(word_index)
print(total_words)

training_sequences = tokenizer.texts_to_sequences(questions)
max_sequence_len = max([len(x) for x in training_sequences])

model = keras.models.load_model('ChatBot.h5')
print('start chatting')
while True: 
    inputvalue = input().lower().replace("?","")
    inputlist = []
    inputlist.append(inputvalue)
    sequences = tokenizer.texts_to_sequences(inputlist)
    padded = pad_sequences(sequences, maxlen=max_sequence_len, padding ="pre", truncating= "post")
    result = np.argmax(model.predict(padded))
    for x,y in dictlabel.items(): 
        if result == y:
            for item in datastore:
                if item['Intention'] == x:
                    answer = item['Answers']
                    print(answer)
                    break
