import math
import numpy as np
import random
import time

import pyoram
from pyoram.util.misc import MemorySize
from pyoram.oblivious_storage.tree.path_oram import PathORAM
from pyoram.encrypted_storage.encrypted_block_storage import EncryptedBlockStorage
from pyoram.storage.block_storage_ram import BlockStorageRAM

SAVE_MEMORY_ACCESSES = True

class OramStorage():
   def __init__(self, storage_name, block_size, block_count):
      self.storage_name = storage_name
      self.block_size = block_size
      with PathORAM.setup(storage_name, block_size, block_count, 
      storage_type = 'ram', ignore_existing = True) as f:
          self.stash = f.stash
          self.position_map = f.position_map
          self.key = f.key
      f.raw_storage.tofile(storage_name)
      self.storage = PathORAM(BlockStorageRAM.fromfile(storage_name), f.stash, f.position_map, key = self.key)             
   
   def start_recording(self):
      if (SAVE_MEMORY_ACCESSES):
         self.start_time = time.time()
         self.read_times = []
         self.read_accesses = []
         self.write_times = []
         self.write_accesses = []
         self.record_access = True

   def read_block(self, index):
      block = self.storage.read_block(index)
      return block

   def write_block(self, index, data):
      self.storage.write_block(index, data)

class RamStorage():
   def __init__(self, storage_name, block_size, block_count):
      self.storage_name = storage_name
      self.block_size = block_size
      with EncryptedBlockStorage.setup(storage_name, block_size, block_count, 
      storage_type = 'ram', ignore_existing = True) as f:
         self.key = f.key
      f.raw_storage.tofile(storage_name)
      self.storage = EncryptedBlockStorage(BlockStorageRAM.fromfile(storage_name), key = self.key)
      self.record_access = False
   
   def start_recording(self):
      if (SAVE_MEMORY_ACCESSES):
         self.start_time = time.time()
         self.read_times = []
         self.read_accesses = []
         self.write_times = []
         self.write_accesses = []
         self.record_access = True

   def read_block(self, index):
      current_time = time.time()
      block = self.storage.read_block(index)
      if (SAVE_MEMORY_ACCESSES):
         if (self.record_access):
            self.read_times.append(current_time - self.start_time)
            self.read_accesses.append(index)
      return block

   def write_block(self, index, data):
      current_time = time.time()
      self.storage.write_block(index, data)
      if (SAVE_MEMORY_ACCESSES):
         if (self.record_access):
            self.write_times.append(current_time - self.start_time)
            self.write_accesses.append(index)
   
   # NOT ORIGINAL CODE
   def save(self):
      if (SAVE_MEMORY_ACCESSES):
         read_times = np.array(self.read_times)
         read_accesses = np.array(self.read_accesses)
         write_times = np.array(self.write_times)
         write_accesses = np.array(self.write_accesses)       
         np.savez(self.storage_name.split('.', 1)[0], read_times = read_times, read_accesses = read_accesses, 
         write_times = write_times, write_accesses = write_accesses)
      
class MemoryMap():
   # Random memory map
   def __init__(self, storage, block_offset, bucket_size, bucket_count, elem_size=4, dtype='float32'):
      self.storage = storage
      self.block_size = storage.block_size
      self.elem_size = elem_size
      self.dtype = dtype
      self.block_offset = block_offset
      self.bucket_size = int(bucket_size)
      self.bucket_count = bucket_count

      self.bucket_data = dict() # indexed by physical bucket 
      self.mmap = dict() # assignment of logical to physical buckets
      self.free = set() # set of free buckets

      for i in range(self.bucket_count):
         self.free.add(i)
  
   def write_blocks(self, start_block, h):
      num_blocks = self.bucket_size
      data_size = int(self.block_size / self.elem_size)
      for i in range(num_blocks):
         block_data = h[i * data_size:(i + 1) * data_size]
         self.storage.write_block(start_block + i, block_data.tobytes())

   def read_blocks(self, start_block):
      data_size = int(self.block_size / self.elem_size)
      num_blocks = self.bucket_size
      h = np.zeros(num_blocks * data_size, dtype = self.dtype)
      for i in range(num_blocks):
         block_data = self.storage.read_block(start_block + i)
         h[i * data_size:(i + 1) * data_size] = np.frombuffer(block_data, dtype = self.dtype)
      return h

   def write_physical_bucket(self, bucket, h):
      start_block = bucket * self.bucket_size + self.block_offset
      # store data parameters
      shape = h.shape
      data_pad = int((self.bucket_size * self.block_size - h.nbytes) / self.elem_size)
         
      # flatten data and add zeros if necessary
      h = h.flatten()
      if (data_pad != 0):
         arr = np.zeros(data_pad, dtype = self.dtype)
         h = np.concatenate((h, arr))
      
      self.write_blocks(start_block, h)
      self.bucket_data[bucket] = (shape, data_pad)
     
   def read_physical_bucket(self, bucket):
      start_block = bucket * self.bucket_size + self.block_offset
      h = self.read_blocks(start_block)

      if (bucket in self.bucket_data):
         shape, data_pad = self.bucket_data[bucket]
         if data_pad != 0:
            h = h[:h.size - data_pad]
         h = h.reshape(shape)

      return h      
           
   def assign_bucket(self, bucket_num):
      bucket = random.choice(tuple(self.free))
      self.free.remove(bucket)
      self.mmap[bucket_num] = bucket
      return bucket      

   def pop_bucket(self, bucket_num):
      bucket = self.mmap.pop(bucket_num)
      self.free.add(bucket)
      return bucket
 
   def write_bucket(self, bucket_num, h):
      bucket = self.assign_bucket(bucket_num)
      self.write_physical_bucket(bucket, h)
   
   def read_bucket(self, bucket_num, free=False):
      if (bucket_num in self.mmap):
         if (free):
            bucket = self.pop_bucket(bucket_num)
         else:
            bucket = self.mmap[bucket_num]
      else:
         bucket = self.assign_bucket(bucket_num)
      h = self.read_physical_bucket(bucket)
      return h       
   
   def free_all_blocks(self):
      for i in range(self.bucket_count):
         self.free.add(i)
      self.mmap = dict()

class MemoryTree():
   # similar to ORAM
   def __init__(self, storage_name, block_size, bucket_size, N, elem_size = 4, dtype = 'float32'):
      bucket_size = int(bucket_size)
      block_count = N * bucket_size
      
      # oram tree parameters
      # number of blocks that will be written to in the tree, not the total number of blocks in the tree
      self.num_buckets = N
      # number of levels in the tree
      self.l = int(math.ceil(math.log(N, 2)) - 1)   
      
      # storage parameters
      self.block_size = block_size
      self.bucket_size = bucket_size
      self.elem_size = elem_size
      self.dtype = dtype
      # create storage
      self.storage = RamStorage(storage_name, block_size, block_count)
      # initialize
      self.reset()

   def reset(self):
      self.bucket_data = dict()
      self.map = dict() # maps logical buckets to the physical buckets they are stored in
      self.position = dict()
      for i in range(self.num_buckets):
         self.position[i] = random.randint(0, (2 ** self.l) - 2)
      self.stash = dict()
      self.leaf = None

   def write_blocks(self, start_block, h):
      num_blocks = self.bucket_size
      data_size = int(self.block_size / self.elem_size)
      for i in range(num_blocks):
         block_data = h[i * data_size:(i + 1) * data_size]
         self.storage.write_block(start_block + i, block_data.tobytes())

   def read_blocks(self, start_block):
      data_size = int(self.block_size / self.elem_size)
      num_blocks = self.bucket_size
      h = np.zeros(num_blocks * data_size, dtype = self.dtype)
      for i in range(num_blocks):
         block_data = self.storage.read_block(start_block + i)
         h[i * data_size:(i + 1) * data_size] = np.frombuffer(block_data, dtype = self.dtype)
      return h

   def write_physical_bucket(self, index, bucket, h):
      start_block = bucket * self.bucket_size

      # store data parameters
      shape = h.shape
      data_pad = int((self.bucket_size * self.block_size - h.nbytes) / self.elem_size)

      self.map[index] = bucket # write the correspondence 

      # flatten data and add zeros if necessary
      h = h.flatten()
      if (data_pad != 0):
         arr = np.zeros(data_pad, dtype = self.dtype)
         h = np.concatenate((h, arr))

      self.write_blocks(start_block, h)
      self.bucket_data[bucket] = (shape, data_pad, index)

   def read_physical_bucket(self, bucket):
      start_block = bucket * self.bucket_size
      h = self.read_blocks(start_block)

      index = -1

      if (bucket in self.bucket_data):
         shape, data_pad, index = self.bucket_data.pop(bucket)
         if data_pad != 0:
            h = h[:h.size - data_pad]
         h = h.reshape(shape)

      return index, h

   def parent(self, i):
      return (i - 1) / 2

   def is_parent(self, i, j):
      # returns true if i is a parent of j
      if (i == 0):
         return True
      elif (j == 0):
         return False

      level_i = int(math.ceil(math.log(i, 2)) - 1)
      level_j = int(math.ceil(math.log(j, 2)) - 1)

      x = False
      for k in range(level_j + 1 - level_i):
         if (i == j):
            x = True
         else:
            j = self.parent(j)
      return x

   def read_leaf_path(self, leaf):
      bucket = (2 ** self.l) + leaf - 1
      for i in range(self.l + 1):
         index, h = self.read_physical_bucket(bucket)
         if (index != -1):
            if (index in self.map):
               if (self.map[index] is not None):
                  self.stash[index] = h
            else:
               self.stash[index] = h
         bucket = self.parent(bucket)

   def write_stash(self, leaf):
      bucket = 2 ** self.l + leaf - 1
      for i in range(self.l + 1):
         block = -1
         for index in self.stash:
            leaf = self.position[index]
            leaf = 2 ** self.l + leaf - 1
            if (self.is_parent(bucket, leaf)):
               block = index
               break
         if (block != -1):
            h = self.stash.pop(block)
            self.write_physical_bucket(block, bucket, h)
         bucket = self.parent(bucket)
   
   def read_bucket(self, index):
      # For bucket data to be valid, must have previously been written to 
      if (self.position[index] == None):
         self.stash[index] = np.zeros(self.bucket_size * self.elem_size)
      else:
         leaf = self.position[index]
         self.read_leaf_path(leaf)
      self.position[index] = random.randint(0, (2 ** self.l) - 2)
      feature_data = self.stash.copy()
      self.write_stash(leaf)
      return feature_data

   def write_path(self, index, data):
      for i in data:
         if i in self.map:
            self.map[i] = None
         self.stash[i] = data[i]
      leaf = self.position[index]
      self.read_leaf_path(leaf)
      self.position[index] = random.randint(0, (2 ** self.l) - 2) 
      self.write_stash(leaf)

   def write_bucket(self, index, h):
      self.stash[index] = h
      leaf = self.position[index]
      self.read_leaf_path(leaf)
      self.position[index] = random.randint(0, (2 ** self.l) - 2)
      self.write_stash(leaf)

