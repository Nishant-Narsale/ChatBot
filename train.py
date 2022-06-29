import json
from nltk_actions import tokenize, stem , bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import NeuralNetwork

with open('intents.json','r') as f:
    intents = json.load(f)

# print(intents)
all_words = []
tag_and_patterns = []
tags = []

for intent in intents['intents']:

    tags.append(intent['tag'])

    tag_patterns = [[],intent['tag']]
    for j in intent['patterns']:
        tokenized_words = tokenize(j)
        all_words.extend(tokenized_words)
        tag_patterns[0].append(tokenized_words)
    tag_and_patterns.append(tuple(tag_patterns))

    for l in intent['responses']:
        tokenized_words = tokenize(l)
        all_words.extend(tokenized_words)

ignore = ['?','!','.',',','-',':',')']
all_words = [stem(w) for w in all_words if w not in ignore]
tags = list(sorted(set(tags)))
all_words = list(sorted(set(all_words)))

# print(all_words)
# print(tags)

X_train = []
y_train = []
for (tokenized_sentences, tag) in tag_and_patterns:
    tag_index = tags.index(tag)
    for sentence in tokenized_sentences:
        bag = bag_of_words(sentence, all_words)
        X_train.append(bag)
        y_train.append(tag_index)

# print(X_train)
# print(y_train)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=2, num_workers=1)

# device = cuda if gpu available else it's cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNetwork(input_size, hidden_size, output_size)

# loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss = {loss.item():.4f}')