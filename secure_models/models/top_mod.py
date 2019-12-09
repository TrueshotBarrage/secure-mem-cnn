import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

import numpy as np
import os
import sys
import time

from manager_mod import *

PATH = 'model/alexnet'

on_chip_storage = torch.Tensor()

class Alexnet(nn.Module):
   def __init__(self):
      super(Alexnet, self).__init__()
      self.features = nn.Sequential(
         nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2),
         nn.Conv2d(64, 192, kernel_size=5, padding=2),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2),
         nn.Conv2d(192, 384, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.Conv2d(384, 256, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.Conv2d(256, 256, kernel_size=3, padding=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2, stride=2),
      )
      self.classifier = nn.Linear(256, 10) # 10 --> CIFAR-10 dataset

   def forward(self, x):
      x = self.features(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      return x

   def classify(self, x):
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      return x 

   def dataview(self, x):
      return x.view(-1, 3, 32, 32)

# Just for the training portion. 
# Called by train_model()
def train(model, device, train_loader, optimizer, criterion):
   model.train()
   train_loss = 0
   correct = 0
   for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      train_loss += loss.item()
      loss.backward()
      optimizer.step()
      pred = output.argmax(dim = 1, keepdim = True)
      correct += pred.eq(target.view_as(pred)).sum().item()
   
   train_loss /= len(train_loader.dataset)

   print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
      .format(train_loss, correct, len(train_loader.dataset), 
      100. * correct / len(train_loader.dataset)))

# Just for the testing portion. 
# Called by train_model()
def test(model, device, test_loader, criterion):
   model.eval()
   test_loss = 0
   correct = 0
   with torch.no_grad():
      for data, target in test_loader:
         data, target = data.to(device), target.to(device)
         output = model(data)
         test_loss += criterion(output, target).item()
         pred = output.argmax(dim = 1, keepdim = True)
         correct += pred.eq(target.view_as(pred)).sum().item()

   test_loss /= len(test_loader.dataset)
 
   print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
      .format(test_loss, correct, len(test_loader.dataset), 
      100. * correct / len(test_loader.dataset)))

# Used to train and test the model, using pytorch API
# The model is saved to PATH: "model/alexnet"
def train_model(train_loader, test_loader, device, epochs):
   model = Alexnet()
   optimizer =  optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

   criterion = nn.CrossEntropyLoss()
   for epoch in range(1, epochs + 1):
      print('Epoch: ' + str(epoch))
      train(model, device, train_loader, optimizer, criterion)
      test(model, device, test_loader, criterion)

   torch.save(model, PATH)
   return model

# For each layer of the neural network, 
# see if weights/bias can be written (i.e. 
# the layer is a convolution layer, not 
# a ReLU or pooling layer).
def write_model(model, manager):
   for i, layer in enumerate(model.features):
      try:
         weights = layer.weight
         bias = layer.bias 
         manager.write_weights(i, weights)
         manager.write_bias(i, bias)
      except:
         if (i == 0) or (i == 3) or (i == 6) or (i == 8) or (i == 10):
            raise

def compute_layer(layer, index, x, manager = None):
   if (manager is None):
      return layer(x)
   else:
      try:
         # Reads all the necessary data
         weights = manager.read_layer_weights(index)
         M = weights[1][1] # ( weights, (num_filters, M), (depth, N) )
         N = weights[2][1]
         num_filters = weights[1][0]
         depth = weights[2][0]
         
         for i in range(int(num_filters / M)):
            for j in range(int(depth / N)):
               weights = manager.read_layer_weights(index, i, j)
               weights = torch.from_numpy(weights[0])
               bias = torch.from_numpy(manager.read_layer_bias(index, i))

               input_feature_map = torch.chunk(x, int(depth / N), dim = 1)[j]
               partial_input = input_feature_map[j]

               if j == 0:
                  conv = F.conv2d(partial_input, weights, bias = bias, \
                     stride = layer.stride, padding = layer.padding)
               else:
                  conv = conv + F.conv2d(partial_input, weights, bias = bias, \
                     stride = layer.stride, padding = layer.padding)
            
            if i == 0:
               result = conv
            else:
               result = torch.cat((result, conv), dim = 1)
         
         return result
      except:
         if (index == 0) or (index == 3) or (index == 6) or (index == 8) or (index == 10):
            raise
         print("rip")
         return layer(x)

def predict(model, test_loader, image, manager):
   data, target = test_loader.dataset[image]
   data = model.dataview(data)

   # Writes all the data, separate into chunks (blocks),
   # to the specified API-given storage manager (i.e. EncryptedBlockStorage)
   manager.write_data(data.numpy())

   # For each layer (Conv2d, ReLU, pool, etc.) 
   # of the given CNN (AlexNet):
   for index, layer in enumerate(model.features):
      on_chip_storage = torch.from_numpy(manager.read_data()) # IFB
      on_chip_storage = compute_layer(layer, index, on_chip_storage, manager) # Conv
      manager.write_data(on_chip_storage.detach().numpy()) # OFB

   on_chip_storage = torch.from_numpy(manager.read_data())
   output = model.classify(on_chip_storage)
   pred = output.argmax(dim = 1, keepdim = True)

   print(('Image {} --> Prediction: ' + str(pred.item())
      + '   Target: ' + str(target)).format(image + 1))

# Phase one
def setup(model, test_loader, images, manager):
   write_model(model, manager)
   
   for image in images:
      predict(model, test_loader, image, manager)

def main():
   device = torch.device("cpu")

   train_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('../data', train=True, download=True, 
      transform=transforms.Compose([transforms.ToTensor(), 
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])), batch_size=64, shuffle=True)

   test_loader = torch.utils.data.DataLoader(
   	datasets.CIFAR10('../data', train=False, transform=transforms.Compose
       ([transforms.ToTensor(), transforms.Normalize
       (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
        batch_size=1000, shuffle=True)    

   epochs = 5

   if (sys.argv[1] == 'train'):
      model = train_model(train_loader, test_loader, device, epochs)
   else:
      model = torch.load(PATH)
   
   images = range(10) # Load 25 images to guess

   manager = Manager()

   setup(model, test_loader, images, manager)

if __name__ == '__main__':
   main()