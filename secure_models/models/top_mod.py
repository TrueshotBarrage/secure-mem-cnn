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

PATH = 'model/hello/alexnet'

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
def write_model(model, storage):
   for i, layer in enumerate(model.features):
      try:
         weights = layer.weight
         bias = layer.bias 
         storage.write_weights(i, weights)
         storage.write_bias(i, bias)
      except:
         pass

def compute_layer(layer, index, x, storage = None):
   if (storage is None):
      return layer(x)
   else:
      try:
         iter_index = 0
         done = False

         while not done:
            weights = storage.read_layer_weights(index, iter_index)
            done = weights[1]
            num_filters = weights[0].shape[0]
            weights = torch.from_numpy(weights[0])

            bias = torch.from_numpy(storage.read_layer_bias(index, iter_index, num_filters))

            # First time iterating through
            if iter_index is 0:
               result = F.conv2d(x, weights, bias = bias, 
               stride = layer.stride, padding = layer.padding)
            else:
               result = torch.cat((result, F.conv2d(x, weights, bias = bias, 
               stride = layer.stride, padding = layer.padding)), dim = 0)
            
            iter_index += 1
         
         return result
      except:
         print("Bad code")
         return layer(x)

def predict(model, test_loader, image, storage):
   data, target = test_loader.dataset[image]
   data = model.dataview(data)

   # Writes all the data, separate into chunks (blocks),
   # to the specified API-given storage (i.e. EncryptedBlockStorage)
   # For example: self.storage (EBS) = block data
   storage.write_data(data.numpy())

   # For each layer (Conv2d, ReLU, pool, etc.) 
   # of the given CNN (AlexNet):
   for index, layer in enumerate(model.features):
      x = torch.from_numpy(storage.read_data()) # Input feature block
      h = compute_layer(layer, index, x, storage) # Convolution
      storage.write_data(h.detach().numpy()) # Output feature block, to be next input

   x = torch.from_numpy(storage.read_data())
   output = model.classify(x)
   pred = output.argmax(dim = 1, keepdim = True)

   print(('Image {} --> Prediction: ' + str(pred.item())
      + '   Target: ' + str(target)).format(image + 1))

# Phase one
def setup(model, test_loader, images, storage):
   write_model(model, storage)
   
   for image in images:
      predict(model, test_loader, image, storage)

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

   storage = Manager(storage_name = "heap.bin")

   setup(model, test_loader, images, storage)

if __name__ == '__main__':
   main()