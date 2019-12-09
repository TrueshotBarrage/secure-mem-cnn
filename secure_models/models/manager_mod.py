import torch
import math
import numpy as np
import pyoram
from pyoram.encrypted_storage.encrypted_block_storage import EncryptedBlockStorage
from pyoram.storage.block_storage_ram import BlockStorageRAM

M = 2 # Precondition: Must be a factor of the # of weights for now
N = 2 # make dict, layerwise
      
class Manager():
   def __init__(self):
      self.block_size = 1024
      self.block_count = 10000
      self.start_block = 0
      self.elem_size = 4
      self.weights = dict()
      self.biases = dict()
      self.on_chip_memory = Storage("onchip.bin", self.block_size, self.block_count)
      self.off_chip_memory = Storage("offchip.bin", self.block_size, self.block_count)

   def write_blocks(self, start_block, num_blocks, h):
      data_size = int(self.block_size / self.elem_size) # constant: 256
      for i in range(num_blocks):
         # 0 ~ 255, 256 ~ 511, and so on
         # h iterates over (data_size * num_blocks) entries 
         block_data = h[i * data_size : (i + 1) * data_size]

         # write_block(index, data) so first iteration is 
         # write_block(0, data that corresponds to block index 0)
         self.off_chip_memory.write_block(start_block + i, block_data.tobytes())

   def read_blocks(self, start_block, num_blocks):
      data_size = int(self.block_size / self.elem_size) # constant: 256
      h = np.zeros(num_blocks * data_size, dtype = "float32")

      for i in range(num_blocks):
         block_data = self.off_chip_memory.read_block(start_block + i)
         h[i * data_size : (i + 1) * data_size] = np.frombuffer(
            block_data, dtype = "float32")
      return h
   
   # h represents weights or bias, depending on the 
   # method calling it; returned value holds no actual data, 
   # but specs for the param, including data_pad (which is an int)
   def write_layer_param(self, h, start_block):
      shape = h.shape
      # eventually implement partial convs with partial filters (remove shape[1])
      fetch_size = shape[2] * shape[3] * shape[1] * M
      num_blocks_per_fetch = int(math.ceil(
         fetch_size * self.elem_size / float(self.block_size)))
      num_blocks = int(num_blocks_per_fetch * shape[0] / float(M))

      data_pad = int((num_blocks_per_fetch * self.block_size \
      - fetch_size * self.elem_size) / self.elem_size)
      for i in range(int(shape[0] / M)):
         data = h[i * M : (i+1) * M].flatten()
         if data_pad != 0:
            arr = np.zeros(data_pad, dtype = "float32")
            data = np.concatenate((data, arr))
         
         self.write_blocks(start_block + i * num_blocks_per_fetch, \
            num_blocks_per_fetch, data)
      return (start_block, num_blocks, shape, data_pad)
   
   # "data" holds no actual data, but specs for the param, 
   # including data_pad (which is an int)
   def read_layer_param(self, data, i = 0):
      (start_block, num_blocks, shape, data_pad) = data
      num_blocks_per_fetch = int(num_blocks * M / float(shape[0]))

      h = self.read_blocks(start_block + i * num_blocks_per_fetch, \
         num_blocks_per_fetch)
      
      if data_pad != 0:
         h = h[: h.size - data_pad]
      
      h = h.reshape((M, shape[1], shape[2], shape[3]))
      return (h, (shape[0], M))

      # if iter_index is not 0:
      #    start_block += iter_index * ON_CHIP_MEMORY_SIZE - 1

      # # Only works for weight filters *
      # num_filters = int((self.block_size / self.elem_size * ON_CHIP_MEMORY_SIZE) \
      #    / (shape[1] * shape[2] * shape[3]))
      # leftover = int((self.block_size / self.elem_size * ON_CHIP_MEMORY_SIZE) \
      #    % (shape[1] * shape[2] * shape[3]))
      # new_shape = (num_filters, shape[1], shape[2], shape[3])
      
      # done = True if iter_index is (iterations - 1) else False
      
      # if not done:
      #    h = self.read_blocks(start_block, ON_CHIP_MEMORY_SIZE)
      #    h = h[: h.size - leftover]
      # else: # Last iteration
      #    remaining = num_blocks % ON_CHIP_MEMORY_SIZE
      #    if remaining is not 0:
      #       h = self.read_blocks(start_block, remaining)
      #    else: # Edge case where remaining is zero or seven
      #       h = self.read_blocks(start_block, ON_CHIP_MEMORY_SIZE)

      #    if data_pad != 0:
      #       h = h[: h.size - data_pad]

      # h = h.reshape(new_shape)
      # return (h, done)

   def write_weights(self, layer, weights):
      h = weights.detach().numpy()
      start_block = self.start_block
      shape = h.shape
      # eventually implement partial convs with partial filters (remove shape[1])
      fetch_size = shape[2] * shape[3] * shape[1] * M
      num_blocks_per_fetch = int(math.ceil(
         fetch_size * self.elem_size / float(self.block_size)))
      num_blocks = int(num_blocks_per_fetch * shape[0] / float(M))

      data_pad = int((num_blocks_per_fetch * self.block_size \
      - fetch_size * self.elem_size) / self.elem_size)
      for i in range(int(shape[0] / M)):
         data = h[i * M : (i+1) * M].flatten()
         if data_pad != 0:
            arr = np.zeros(data_pad, dtype = "float32")
            data = np.concatenate((data, arr))
         
         self.write_blocks(start_block + i * num_blocks_per_fetch, \
            num_blocks_per_fetch, data)

      self.start_block += num_blocks
      self.weights[layer] = (start_block, num_blocks, shape, data_pad)

   def write_bias(self, layer, bias):
      h = bias.detach().numpy()
      start_block = self.start_block
      shape = h.shape

      num_blocks = int(math.ceil(h.nbytes / float(self.block_size)))

      data_pad = int((num_blocks * self.block_size - h.nbytes) / self.elem_size)

      h = h.flatten()
      if (data_pad != 0):
         arr = np.zeros(data_pad, dtype = "float32")
         h = np.concatenate((h, arr))
         
      self.write_blocks(start_block, num_blocks, h)

      self.start_block += num_blocks
      self.biases[layer] = (start_block, num_blocks, shape, data_pad)

   def read_layer_weights(self, layer, i = 0):
      (start_block, num_blocks, shape, data_pad) = self.weights[layer]
      num_blocks_per_fetch = int(num_blocks * M / float(shape[0]))

      h = self.read_blocks(start_block + i * num_blocks_per_fetch, \
         num_blocks_per_fetch)
      
      if data_pad != 0:
         h = h[: h.size - data_pad]
      
      h = h.reshape((M, shape[1], shape[2], shape[3]))
      return (h, (shape[0], M))
 
   def read_layer_bias(self, layer, i = 0):
      (start_block, num_blocks, shape, data_pad) = self.biases[layer]
      h = self.read_blocks(start_block, num_blocks)
      
      if data_pad != 0:
         h = h[: h.size - data_pad]
      
      h = h[i * M : (i+1) * M]
      h = h.reshape((M,))
      return h
   
   def write_data(self, h):
      self.array_shape = h.shape
      # eventually implement partial convs with partial filters (add N)
      image_size = self.array_shape[2] * self.array_shape[3] * self.array_shape[1]
      self.num_blocks = int(math.ceil(
         image_size * self.elem_size / float(self.block_size)))

      self.pad = int((self.num_blocks * self.block_size \
      - image_size * self.elem_size) / self.elem_size)
      
      h = h.flatten()
      if (self.pad != 0):
         arr = np.zeros(self.pad, dtype = "float32")
         h = np.concatenate((h, arr))
      
      self.write_blocks(self.start_block, self.num_blocks, h)

   def read_data(self):     
      h = self.read_blocks(self.start_block, self.num_blocks)
      if self.pad != 0:
          h = h[:h.size - self.pad]

      h = h.reshape(self.array_shape)
      return h

class Storage():
   def __init__(self, storage_name, block_size, block_count):
      self.storage_name = storage_name
      self.block_size = block_size
      with EncryptedBlockStorage.setup(storage_name, block_size, block_count, 
      storage_type = 'ram', ignore_existing = True) as f:
         self.key = f.key
      f.raw_storage.tofile(storage_name)
      self.storage = EncryptedBlockStorage(
         BlockStorageRAM.fromfile(storage_name), key = self.key)
      
   def read_block(self, index):
      block = self.storage.read_block(index)
      return block
   
   def write_block(self, index, data):
      self.storage.write_block(index, data)
