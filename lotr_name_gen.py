# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 22:26:12 2019
LoTR Name Dataset from https://www.kaggle.com/paultimothymooney/lord-of-the-rings-data#lotr_scripts.csv
Based on the tutorial https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
@author: Garnet
"""

# Imports
# Torch stuff
import torch
import torch.nn as nn
# For reading csvs
import csv
# For removing non-ASCII characters
import string
import unicodedata
# For random
import random
# To track how long training takes
import time
import math
# To plot the loss results
import matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
    
# Get a random race, name pair
def randomTrainingPair():
    race = races[random.randint(0, len(races) - 1)]
    name = raceNames[race][random.randint(0, len(raceNames[race]) - 1)]
    return race, name

# Get a one-hot vector for category
def raceTensor(race):
    li = races.index(race)
    tensor = torch.zeros(1, n_races)
    tensor[0][li] = 1
    return tensor

# Get a one-hot matrix for first to last letters (not including EOS)
def inputTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for li in range(len(name)):
        letter = name[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# Get a LongTensor of second letter to EOS for target
def targetTensor(name):
    letter_indexes = [all_letters.find(name[li]) for li in range(1, len(name))]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)

# Create category, input, and target tensors from a random race, name pair
def randomTrainingExample():
    race, name = randomTrainingPair()
    race_tensor = raceTensor(race)
    input_name_tensor = inputTensor(name)
    target_name_tensor = targetTensor(name)
    return race_tensor, input_name_tensor, target_name_tensor

def train(race_tensor, input_name_tensor, target_name_tensor):
    target_name_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()
    
    rnn.zero_grad()
    
    loss = 0
    
    for i in range(input_name_tensor.size(0)):
        output, hidden = rnn(race_tensor, input_name_tensor[i], hidden)
        l = criterion(output, target_name_tensor[i])
        loss += l
    
    loss.backward()
    
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    
    return output, loss.item() / input_name_tensor.size(0)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m,s)

    
def sample(race, start_letter='A'):
    with torch.no_grad():   # no need to track history in sampling
        race_tensor = raceTensor(race)
        input_ = inputTensor(start_letter)
        hidden = rnn.initHidden()
        
        output_name = start_letter
        
        for i in range(max_length):
            output, hidden = rnn(race_tensor, input_[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:   # EOS marker
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input_ = inputTensor(letter)
    return output_name

def samples(race, start_letters='ABC'):
    for start_letter in start_letters:
        name = sample(race, start_letter)
        print("New name? ", name not in raceNames[race], ":", name)
            

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(n_races + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_races + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, race, input_, hidden):
        input_combined = torch.cat((race, input_, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
        
        
    

if __name__ == '__main__':
    max_length = 0
    with open('lotr_characters.csv') as csvfile:
        lotrReader = csv.reader(csvfile, delimiter=',')
        next(lotrReader, None)
        raceNames = {}
        # birth,death,gender,hair,height,name,race,realm,spouse
        for birth, death, gender, hair, height, name, race, realm, spouse in lotrReader:
            if race == 'Hobbit':
                race = 'Hobbits'
            if race in ['Men', 'Elves', 'Hobbits'] and race != '':
                if '(' in name:
                    name = name[:name.find('(')-1]
                name = unicodeToAscii(name)
                if len(name) > max_length:
                    max_length = len(name)
                if race not in raceNames:
                    raceNames[race] = [name]
                else:
                    raceNames[race].append(name)
    
    races = []
    for race in raceNames:
        races.append(race)
    n_races = len(races)
    print(races)
        
    for race, names in raceNames.items():
        print(race, 'Names: ', len(names), '\n', names, '\n')
    print("\n---------------------------------------\n")
    
    #################################################################
    # Train the network
    #################################################################
    criterion = nn.NLLLoss()
    learning_rate = 0.005
    
    rnn = RNN(n_letters, 128, n_letters)
    
    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    
    start = time.time()
    
    for i in range(1, n_iters  + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss
        
        if i % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), i, i / n_iters * 100, loss))
            
            if i % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
            
    plt.figure()
    plt.plot(all_losses)
    
    #################################################################
    # Sample the network
    #################################################################
    print("Elves")
    samples('Elves', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    print("Humans")
    samples('Men', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    print("Hobbits")
    samples('Hobbits', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            
            
            
            
            
            
            
            
            
            
            
            

    
    
    
    
    
    
    