import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

import numpy as np

import os
import sys
import time

from simple_manager import SimpleModelManager
from random_map import RandomMapModel
from tree_manager import OramModel

import pdb

storage_name = "heap.bin"
storage_oram_name = "heap_oram.bin"
block_size = 1024
block_count = 10000

num_classes = 10

PATH="model/alexnet"

class Alexnet(torch.nn.Module):
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
      self.classifier = nn.Linear(256, num_classes)

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
      .format(train_loss, correct, len(train_loader.dataset), 100. * correct/len(train_loader.dataset)))
          

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
      .format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def write_model(model, f):
   for i, layer in enumerate(model.features):
      try:
         weights = layer.weight
         bias = layer.bias 
         f.write_weights(i, weights)
         f.write_bias(i, bias)
      except:
         pass

def compute_layer(layer, index, x, f = None):
   if (f is None):
      return layer(x)
   else:
      try:
         weights = torch.from_numpy(f.read_layer_weights(index))
         bias = torch.from_numpy(f.read_layer_bias(index))
         return F.conv2d(x, weights, bias = bias, stride = layer.stride, padding = layer.padding)
      except:
         return layer(x)

def predict_test(model, device, test_loader, image):
   data, target = test_loader.dataset[image]
   data = model.dataview(data)

   x = data
   for i, layer in enumerate(model.features):
      x = compute_layer(layer, i, x)
   
   output = model.classify(x)
   pred = output.argmax(dim = 1, keepdim = True)
 
   print('No Storage		Prediction: ' + str(pred.item()) + '	Target: ' + str(target))
   
def predict_images(model, device, test_loader, images):
   start_time = time.time()
   for image in images:
      predict_test(model, device, test_loader, image)
   end_time = time.time()
   return end_time - start_time

def predict_test_store(model, device, test_loader, image, f):
   data, target = test_loader.dataset[image]
   data = model.dataview(data)

   f.start_recording()

   # Writes all the data, separate into chunks (blocks),
   # to the specified API-given storage (i.e. EncryptedBlockStorage)
   # For example: self.storage (EBS) = block data
   f.write_data(data.numpy())

   # For each layer (Conv2d, ReLU, pool, etc.) 
   # of the given CNN (AlexNet):
   for i, layer in enumerate(model.features):
      x = torch.from_numpy(f.read_data())
      h = compute_layer(layer, i, x, f = f)
      f.write_data(h.detach().numpy())

   f.save()
   x = torch.from_numpy(f.read_data())
   output = model.classify(x)
   pred = output.argmax(dim = 1, keepdim = True)

   print('With Storage ' + f.storage_type + '	Prediction: ' + str(pred.item()) + '	Target: ' + str(target))

def predict_test_store_opt(model, device, test_loader, image, f):
   data, target = test_loader.dataset[image]
   
   data = model.dataview(data)
   x = data
   for i, layer in enumerate(model.features):
      if (len(list(layer.parameters())) == 2):
         # convolution layer
         # Simulate writing and reading of data
         f.write_data(x.detach().numpy())
         x = torch.from_numpy(f.read_data())
         x = compute_layer(layer, i, x, f = f)
      else:
         x = compute_layer(layer, i, x, f = f)
   f.save()
   output = model.classify(x)
   pred = output.argmax(dim = 1, keepdim = True)

   print('With Storage(OPT) ' + f.storage_type + '	Prediction: ' + str(pred.item()) + '	Target: ' + str(target))

def predict_images_store(model, device, test_loader, images, f):
   write_model(model, f)
   
   start_time = time.time()
   for image in images:
      predict_test_store(model, device, test_loader, image, f)
   end_time = time.time()
   return end_time - start_time

def predict_images_store_opt(model, device, test_loader, images, f):
   write_model(model, f)
   
   start_time = time.time()
   for image in images:
      predict_test_store_opt(model, device, test_loader, image, f)
   end_time = time.time()
   return end_time - start_time

def train_model(train_loader, test_loader, device, epochs):
   model = Alexnet()
   optimizer=optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

   criterion = nn.CrossEntropyLoss()
   for epoch in range(1, epochs + 1):
      print('Epoch: ' + str(epoch))
      train(model, device, train_loader, optimizer, criterion)
      test(model, device, test_loader, criterion)

   torch.save(model, PATH)
   return model

def predict_test_random_map(model, test_loader, image, f):
   data, target = test_loader.dataset[image]
   data = model.dataview(data)

   x = f.compute_model(data)
   # print(x)
   output = model.classify(x.float())
   # print(output)
   pred = output.argmax(dim = 1, keepdim = True) # keepdim = True originally
   # print(pred)

   print('With Storage Random Map	Prediction: ' + str(pred.item()) + '	Target: ' + str(target))  

def predict_images_random_map(model, test_loader, images):
   data, target = test_loader.dataset[images[0]]
   data = model.dataview(data)
   f = RandomMapModel(model, data, block_size, block_count)
   # f = OramModel(model, data, block_size, block_count) # OramModel works as intended

   start_time = time.time()
   for image in images:
      predict_test_random_map(model, test_loader, image, f)
   end_time = time.time()
   return end_time - start_time

def predict_test_tree(model, test_loader, image, f):
   data, target = test_loader.dataset[image]
   data = model.dataview(data)
   
   x = f.compute_model(data)
   # print(x)
   output = model.classify(x.float())
   pred = output.argmax(dim = 1, keepdim = True)

   print('With Storage Tree	Prediction: ' + str(pred.item()) + '	Target: ' + str(target))

def predict_images_tree(model, test_loader, images):
   data, target = test_loader.dataset[images[0]]
   data = model.dataview(data)
   f = OramModel(model, data, block_size, block_count)

   start_time = time.time()
   for image in images:
      predict_test_tree(model, test_loader, image, f)
   end_time = time.time()
   return end_time - start_time

def main():
   device = torch.device("cpu")
   
   train_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), 
      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])), batch_size=64, shuffle=True)
   
   test_loader = torch.utils.data.DataLoader(
   	datasets.CIFAR10('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), 
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])), batch_size=1000, shuffle=True)    

   epochs = 5
   
   if (sys.argv[1] == 'train'):
      model = train_model(train_loader, test_loader, device, epochs)
   elif (sys.argv[1] == 'load'):
      model = torch.load(PATH)
   
   images = range(int(sys.argv[2]))

   ramStorage = SimpleModelManager(storage_name, block_size, block_count, 'RAM')
   oramStorage = SimpleModelManager(storage_name, block_size, block_count, 'ORAM')
   
   time = predict_images(model, device, test_loader, images)
   print('Prediction time with no storage:	' + str(time) + ' s')

   time = predict_images_store(model, device, test_loader, images, ramStorage)
   print('Prediction time with RAM storage:	' + str(time) + ' s')

   #time = predict_images_store_opt(model, device, test_loader, images, ramStorage)
   #print('Prediction time with RAM storage opt:	' + str(time) + ' s')

   time = predict_images_random_map(model, test_loader, images)
   print('Prediction time with Random Map storage: 	' + str(time) + ' s')

   time = predict_images_tree(model, test_loader, images)
   print('Prediction time with Tree storage:	' + str(time) + ' s')

   #time = predict_images_store(model, device, test_loader, images, oramStorage)
   #print('Prediction time with ORAM storage:	' + str(time) + ' s')

   #time = predict_images_store_opt(model, device, test_loader, images, oramStorage)
   #print('Prediction time with ORAM storage opt:	' + str(time) + ' s')

if __name__ == '__main__':
   main()
