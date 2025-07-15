# main.py - Model Training for ANN ChatBot

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download required NLTK data (only once needed)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load your intents file
with open("E:/chatbot/intents_dataset_1000.json", encoding="utf-8") as file:
    intents = json.load(file)

# Preprocessing
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lowercase
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Save words and classes for future use
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag + output_row)

# Shuffle and convert to numpy
random.shuffle(training)
training = np.array(training, dtype=float)

# Split input and output
train_x = np.array(list(training[:, 0:len(words)]))
train_y = np.array(list(training[:, len(words):]))

# Model architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save("chatbot_model.h5", hist)
print("Chatbot model trained and saved successfully!")
