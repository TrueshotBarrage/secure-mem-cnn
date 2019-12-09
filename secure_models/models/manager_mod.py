import torch
import math
import numpy as np
import pyoram
from pyoram.encrypted_storage.encrypted_block_storage import EncryptedBlockStorage
from pyoram.storage.block_storage_ram import BlockStorageRAM

# Minimum on-chip memory size for given M & N; maybe different per layer
# (Worst case on-chip memory size for all the layers)
# Number of memory reads & writes for each layer
# (Plot of memory transfer, using mem_addr & time)

# Separate on-chip mem structures for W, IFM, OFM
# Precondition: Must be a factor of the # of weights for now
M = dict()
M[0] = 2
M[1] = 0
M[2] = 0
M[3] = 2
M[4] = 0
M[5] = 0
M[6] = 2
M[7] = 0
M[8] = 2
M[9] = 0
M[10] = 2
M[11] = 0
M[12] = 0

N = dict()
N[0] = 3
N[1] = 0
N[2] = 0
N[3] = 16
N[4] = 0
N[5] = 0
N[6] = 16
N[7] = 0
N[8] = 16
N[9] = 0
N[10] = 16
N[11] = 0
N[12] = 0
      
class Manager():
   def __init__(self):
      self.block_size = 512
      self.block_count = 25600
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

   def write_weights(self, layer, weights):
      h = weights.detach().numpy()
      start_block = self.start_block
      shape = h.shape
      # k^2 * M * N
      fetch_size = shape[2] * shape[3] * N[layer] * M[layer]
      num_blocks_per_fetch = int(math.ceil(
         fetch_size * self.elem_size / float(self.block_size)))
      num_blocks = int(num_blocks_per_fetch * shape[0] * shape[1]
         / (M[layer] * N[layer]))

      data_pad = int((num_blocks_per_fetch * self.block_size \
      - fetch_size * self.elem_size) / self.elem_size)
      # for number of fetches:
      for i in range(int(shape[0] / M[layer])):
         for j in range(int(shape[1] / N[layer])):
            data = h[i*M[layer] : (i+1)*M[layer], j*N[layer] : (j+1)*N[layer]]
            data = data.flatten()
            if data_pad != 0:
               arr = np.zeros(data_pad, dtype = "float32")
               data = np.concatenate((data, arr))
            
            self.write_blocks(start_block + num_blocks_per_fetch \
               * int(shape[1] / N[layer] * i + j), num_blocks_per_fetch, data)

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

   def read_layer_weights(self, layer, i = 0, j = 0):
      (start_block, num_blocks, shape, data_pad) = self.weights[layer]
      num_blocks_M = int(num_blocks * M[layer] / float(shape[0]))
      num_blocks_N = int(num_blocks_M * N[layer] / float(shape[1]))

      h = self.read_blocks(start_block + i * num_blocks_M + j * num_blocks_N, \
         num_blocks_N)
      
      if data_pad != 0:
         h = h[: h.size - data_pad]
      
      h = h.reshape((M[layer], N[layer], shape[2], shape[3]))
      return (h, (shape[0], M[layer]), (shape[1], N[layer]))
 
   def read_layer_bias(self, layer, i = 0):
      (start_block, num_blocks, shape, data_pad) = self.biases[layer]
      h = self.read_blocks(start_block, num_blocks)
      
      if data_pad != 0:
         h = h[: h.size - data_pad]
      
      h = h[i * M[layer] : (i+1) * M[layer]]
      h = h.reshape((M[layer],))
      return h
   
   def write_data(self, h):
      self.array_shape = h.shape
      # eventually implement partial convs with partial filters (add N)
      image_size = self.array_shape[0] * self.array_shape[2] * self.array_shape[3] * self.array_shape[1]
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
