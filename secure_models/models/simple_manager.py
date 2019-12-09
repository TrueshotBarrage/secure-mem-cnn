import torch

import math
import numpy as np

from storage_manager import *
      
class SimpleModelManager():
   def __init__(self, storage_name, block_size, block_count, storage_type = 'RAM', elem_size = 4, dtype = 'float32'):
      if storage_type == 'ORAM':
          self.storage = OramStorage(storage_name, block_size, block_count)
      else:
          self.storage = RamStorage(storage_name, block_size, block_count)

      self.storage_type = storage_type
      self.block_size = block_size
      self.block_count = block_count
      self.start_block = 0
      self.elem_size = elem_size
      self.dtype = dtype
      self.weights = dict()
      self.biases = dict()

   def write_blocks(self, start_block, num_blocks, h):
      data_size = self.block_size / self.elem_size
      for i in range(num_blocks):
         # 0 ~ 255, 256 ~ 511, and so on (depends on data_size)
         block_data = h[i * data_size : (i + 1) * data_size]
         # write_block(index, data) so first iteration is 
         # write_block(0, data that corresponds to block index 0)
         self.storage.write_block(start_block + i, block_data.tobytes())

   def read_blocks(self, start_block, num_blocks):
      data_size = self.block_size / self.elem_size
      h = np.zeros(num_blocks * data_size, dtype = self.dtype)
      for i in range(num_blocks):
         block_data = self.storage.read_block(start_block + i)
         h[i * data_size : (i + 1) * data_size] = np.frombuffer(block_data, dtype = self.dtype)
      return h
   
   # h represents weights or bias, depending on the function calling it
   def write_layer_param(self, h, start_block):
      shape = h.shape
      num_blocks = int(math.ceil(h.nbytes / float(self.block_size)))

      # Pads the data to the worst case size feature block(?)
      data_pad = (num_blocks * self.block_size - h.nbytes) /  self.elem_size
      h = h.flatten()
      if (data_pad != 0):
         arr = np.zeros(data_pad, dtype = self.dtype)
         h = np.concatenate((h, arr))
      
      self.write_blocks(start_block, num_blocks, h)
      return (start_block, num_blocks, shape, data_pad)

   def read_layer_param(self, data):
      (start_block, num_blocks, shape, data_pad) = data
      data_size = int(self.block_size / self.elem_size)
      h = self.read_blocks(start_block, num_blocks)
      
      if data_pad != 0:
          h = h[:h.size - data_pad]

      h = h.reshape(shape)
      return h

   def write_weights(self, layer, weights):
      weights = weights.detach().numpy()
      weight_data = self.write_layer_param(weights, self.start_block)
      self.start_block += weight_data[1]
      self.weights[layer] = weight_data

   def write_bias(self, layer, bias):
      bias = bias.detach().numpy()
      bias_data = self.write_layer_param(bias, self.start_block)
      self.start_block += bias_data[1]
      self.biases[layer] = bias_data

   def read_layer_weights(self, layer):
      return self.read_layer_param(self.weights[layer])
 
   def read_layer_bias(self, layer):
      return self.read_layer_param(self.biases[layer])
   
   def write_data(self, h):
      self.array_shape = h.shape
      self.num_blocks = int(math.ceil(h.nbytes / float(self.block_size)))
      self.pad = (self.num_blocks * self.block_size - h.nbytes) / self.elem_size     

      h = h.flatten()
      if (self.pad != 0):
         arr = np.zeros(self.pad, dtype = self.dtype)
         h = np.concatenate((h, arr))
     
      self.write_blocks(self.start_block, self.num_blocks, h) 

   def read_data(self):
      data_size = int(self.block_size / self.elem_size)
     
      h = self.read_blocks(self.start_block, self.num_blocks)
      if self.pad != 0:
          h = h[:h.size - self.pad]

      h = h.reshape(self.array_shape)
      return h    

   def start_recording(self):
      # start recording memory accesses
      if (self.storage_type == 'RAM'):
         self.storage.start_recording()

   def save(self):
      # save memory accesses
      if self.storage_type == 'RAM':
         self.storage.save()     
