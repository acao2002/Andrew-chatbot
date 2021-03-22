# Andrew's Chatbot 

Too tired of breaking the ice yourself? Got too many people wanting to get to know you? That will no longer be a problem when you have a chatbot that can answer questions about yourself. Hence, I have decided to build a Machine Learning model to train my bot so as it can talk to people as if I am answering them myself. 

The project uses **NLP techniques** such as ***Tokenization and padding word sequences*** to formulate strings of questions into same-sized matrix inputs for training. 

The model utilizes an **embedding layer** and a **lstm layer**, with a **softmax activation function** to categorize the intents of the questions. From the identified intents, the chatbot picks the correlated answer to reply to the questions. 

## Technologies 

1. Machine Learning(ML)
2. Natural Language Processing(NLP) 
3. Python 3 


## Install

The project needs Python 3.6+ with tensorflow and numpy installed 

```
git clone https://github.com/acao2002/An-chatbot.git

```

## Launch 

To launch the chatbot, run 
```
python ChatBotImplementation.py
```
or just run the ChatBotImplementation.py file in your IDE 

## Talk to my bot

Start typing your question and talk to me !

```
start chatting
you: what is your name?
An: My name is An Cao
you: are you dating?
An: I am single
you: where do u live?
An: I am from HCM city, Vietnam
you: school u going to ?
An: Vanderbilt University
you: you play sports?
An: I used to run cross-country and play ultimate frisbee. Now I mainly workout in the gym and I am trying boxing
```

## License

MIT

