import torch
import torch.nn.functional as F

import numpy
import random
import math
import time

from storage_manager import *
from model_parser import *
from parameter_store import *
from scheduler import *

import pdb

FEATURE_STORAGE_NAME = 'features.bin'
PARAM_STORAGE_NAME = 'parameters.bin'
MAX_ON_CHIP_MEMORY = 50 * 1024
BLOCKS_PER_SCHEDULE = 2
MAX_KERNELS = 8
SCALE_FACTOR = 2

class Scheduler():
   def __init__(self, num_layers, layer_data, block_size, elem_size = 4):
      # layer_data is a dictionary containing the shapes of the input feature
      # map for every layer. The shape of the output to the network is stored as 
      # the data for layer num_layers. The shape of each layer should be (1, N, H, W)
      # layer_type is either 'CONVOLUTION' or 'SOFTWARE'.

      self.num_layers = num_layers
      self.elem_size = elem_size

      # Creates arrays filled with zeroes
      self.ch_size = np.zeros(self.num_layers + 1, dtype='int32')
      self.num_chs = np.zeros(self.num_layers + 1, dtype='float32')
      
      # Populates ch_size and num_chs with the appropriate layer data
      for i in range(self.num_layers + 1):
         self.ch_size[i] = layer_data[i][2] * layer_data[i][3]
         self.num_chs[i] = layer_data[i][1]

      self.block_size = block_size
      self.min_bucket_size = np.max(self.ch_size) * self.elem_size / self.block_size
      self.max_bucket_size = (MAX_ON_CHIP_MEMORY / block_size) / BLOCKS_PER_SCHEDULE

      if (self.max_bucket_size < self.min_bucket_size):
         print('[ERROR] CANNOT DETERMINE STORAGE PARAMETERS')

   def num_kernels(self, bucket_size):
      b = bucket_size / self.elem_size # number of elements that can fit in bucket
      y = np.ceil(self.num_chs / (b / self.ch_size))
      return int(np.max(y))

   def set_storage_parameters(self):
      min_memory = 0
      min_kernels = 0
      min_bucket = 0
      for i in range(int(self.min_bucket_size), int(self.max_bucket_size) + 1):
         bucket_size = i * self.block_size
         kernels = self.num_kernels(bucket_size)
         max_blocks = MAX_ON_CHIP_MEMORY / bucket_size
         j = max_blocks / 2
         k = max_blocks - j
         memory = (kernels + j - 1) / j * (kernels + k - 1) / k * bucket_size
         if (min_memory == 0 or memory < min_memory):
            min_memory = memory
            min_kernels = kernels
            min_bucket = bucket_size

      self.num_blocks = SCALE_FACTOR * min_kernels
      self.bucket_size = min_bucket # set in number of bytes

      return (self.num_blocks, self.bucket_size)

   def set_layer_schedule(self):
      max_blocks = MAX_ON_CHIP_MEMORY / self.bucket_size
      return schedule(self.num_blocks, max_blocks, MAX_KERNELS)

   def set_schedule(self, layer_types):
      # Returns dictionary of schedule
      # schedule is stored as ((jobs, order), append)
      self.schedule = dict()
      conv_layer = -1
      for i in range(self.num_layers):
         append_layers = []
         if (layer_types[i] == 'CONVOLUTION'):
            schedule = self.set_layer_schedule()
            self.schedule[i] = (schedule, append_layers)
            conv_layer = i
         elif (layer_types[i] == 'SOFTWARE'):
            self.schedule[conv_layer][1].append(i)

   def get_layer_schedule(self, layer):
      # Returns software layers that must be appended to the current layer
      return self.schedule[layer]

   def get_block_data(self, layer_types):
      # Returns valid block data and channels per bucket
      blocks = np.where(self.num_chs < self.num_blocks, self.num_chs, self.num_blocks) # originally <
      chs = np.ceil(self.num_chs / blocks).astype('int32')
      valid_blocks = (self.num_chs / chs).astype('int32')
      conv_blocks = 0
      conv_chs = 0

      # Determine data for software layers
      for i in range(self.num_layers):
         if layer_types[i] == 'CONVOLUTION':
            conv_blocks = valid_blocks[i + 1]
            conv_chs = chs[i + 1]
         elif layer_types[i] == 'SOFTWARE':
            valid_blocks[i + 1] = conv_blocks
            chs[i + 1] = conv_chs
       
      return valid_blocks, chs
  
class RandomMapModel():
   def __init__(self, model, image, block_size, block_count):
      # Parse model to get model data
      self.net = ModelParser(model, image)

      # Create manager and set storage parameters
      self.scheduler = Scheduler(self.net.num_layers, self.net.layer_data, block_size)
      print('[INFO] DETERMINING FEATURE STORAGE PARAMETERS')
      feature_blocks, feature_size = self.scheduler.set_storage_parameters()
      print('[INFO] NUMBER OF FEATURE BLOCKS PER LAYER ' + str(feature_blocks))
      print('[INFO] SIZE OF FEATURE BLOCKS (NUMBER OF MEMORY BLOCKS)	' + str(feature_size / block_size))
      self.scheduler.set_schedule(self.net.layer_types)
      print('[INFO] SET SCHEDULE')
      self.feature_blocks = feature_blocks

      # Write parameters
      valid_blocks, chs = self.scheduler.get_block_data(self.net.layer_types)
      self.params = ParameterStore(model)
      num_blocks = self.params.get_block_count(model, self.net.layer_types, block_size, 
      valid_blocks, chs, pad = True)
      param_storage = RamStorage(PARAM_STORAGE_NAME, block_size, num_blocks) # normally RamStorage
      self.params.set_storage(param_storage)
      print('[INFO] SET UP PARAMETER STORAGE WITH ' + str(self.params.num_buckets) 
      + ' BUCKETS WITH ' + str(self.params.bucket_size) + ' BLOCKS')
      self.params.write_parameters(model, self.net.layer_types, valid_blocks, chs, num_blocks = feature_blocks)
      print('[INFO] FINISHED WRITING WEIGHTS')

      # Create feature storage
      feature_storage = RamStorage(FEATURE_STORAGE_NAME, block_size, block_count) # normally RamStorage
      # Originally block_size = 0, 3 * feature_blocks instead of 2 * feature_blocks
      self.feature_map = MemoryMap(feature_storage, block_size, feature_size / block_size, 3 * feature_blocks)
      print('[INFO] SET UP FEATURE STORAGE')

   def get_weight_data(self, layer, tasks, valid):
      # given set of tasks, returns dictionary of weights necessary for computation
      # dictionary is indexed by task
      weights = dict()
      for task in tasks:
         x, y = task
         if (x != -1):
            weights[(x, y)] = torch.from_numpy(self.params.get_weights(layer, task))
      return weights

   def get_bias_data(self, layer, tasks, valid, channels_bucket, output_table):
      # given set of tasks return the biases needed in a dictionary
      bias = dict()
      for task in tasks:
         x, y = task
         if (y < valid[1]):
            if (y not in bias and x == -1):
               _, out_ch = channels_bucket
               bias_data = self.params.get_bias(layer, y)
               bias[y] = torch.from_numpy(bias_data.reshape(1, out_ch, 1, 1))
      return bias 

   def get_feature_data(self, tasks, indices):
      # given set of tasks return the feature data necessary in a dictionary
      feature_data = dict()
      for task in tasks: 
          x, y = task
          i = x + indices[0]
          j = y + indices[1]
          if (i not in feature_data):
              feature_data[i] = torch.from_numpy(self.feature_map.read_bucket(i, free=True))
          if (j not in feature_data):
              feature_data[j] = torch.from_numpy(self.feature_map.read_bucket(j, free=True))

      return feature_data

   def write_feature_data(self, feature_data):
       for i in feature_data:
           h = feature_data[i].detach().numpy()
           self.feature_map.write_bucket(i, h)

   def compute_software_layers(self, layers, feature_data, feature_blocks, indices):
       # given list of feature_blocks that are done, will compute layers
       for block in feature_blocks:
          index = block + indices[1]
          for layer in layers:
             feature_data[index] = self.net.compute_task(layer, feature_data[index])

   def compute_tasks(self, layer, tasks, feature_data, weights, bias, valid, indices, output_table):
       # given set of tasks, compute the tasks
       for task in tasks:
          x, y = task
          i = x + indices[0]
          j = y + indices[1]
          if (x < valid[0] and y < valid[1]):
             if (output_table[y] == 0):
                if (x == -1):
                   # add bias task
                   feature_data[j] = bias[y]
                else:
                    feature_data[j] = self.net.compute_task(layer, feature_data[i], weights[task])
                output_table[y] = 1
             else:
                if (x == -1):
                   # add bias task
                   feature_data[j] += bias[y]
                else: 
                   feature_data[j] = self.net.compute_task(layer, feature_data[i], weights[task])
                output_table[y] += 1
       # check for finished blocks
       blocks_finished = []
       for y in range(valid[1]):
          if (output_table[y] == (valid[0] + 1)):
             blocks_finished.append(y)
             output_table[y] += 1
       return blocks_finished
    
   def compute_job(self, layer, tasks, software_layers, indices, valid_blocks, chs, output_table):
       # read in feature data
       feature_data = self.get_feature_data(tasks, indices)
       # read in weights
       weights = self.get_weight_data(layer, tasks, valid_blocks)
       # read in bias data
       bias = self.get_bias_data(layer, tasks, valid_blocks, chs, output_table)
       # compute tasks
       blocks_finished = self.compute_tasks(layer, tasks, feature_data, 
       weights, bias, valid_blocks, indices, output_table)
       # compute software layers
       self.compute_software_layers(software_layers, feature_data, blocks_finished, indices) 
       # write back feature data
       self.write_feature_data(feature_data)

   def compute_layer(self, layer, indices, valid_blocks, chs):
       in_blocks, out_blocks = valid_blocks
       output_table = np.zeros(out_blocks)
       (schedule, software_layers) = self.scheduler.get_layer_schedule(layer)
       
       jobs, order = schedule
       for i in order:
          tasks = jobs[i]
          self.compute_job(layer, tasks, software_layers, indices, valid_blocks, chs, output_table)

       return (indices[1], indices[0])

   def compute_layers(self, indices, valid_blocks, chs):
       layer_types = self.net.get_layer_types()

       for i in range(self.net.num_layers):
          #print('[INFO] COMPUTING LAYER ' + str(i))
          layer_type = layer_types[i]
          if (layer_type == 'CONVOLUTION'):
             valid = (valid_blocks[i], valid_blocks[i + 1])
             channels_per_bucket = (chs[i], chs[i + 1])
             indices = self.compute_layer(i, indices, valid, channels_per_bucket)

       return indices[0]

   def write_input(self, image, valid_blocks, chs_bucket):
       data = dict()
       # Writes one channel to one bucket
       for i in range(valid_blocks):
          h = image[:, i * chs_bucket:(i + 1) * chs_bucket, :, :]
          data[i] = h

       # Write zeros to all other blocks
       for i in range(valid_blocks, 2 * self.feature_blocks):
          h = torch.from_numpy(np.zeros(1, dtype='float32'))
          data[i] = h

       self.write_feature_data(data)

   def read_output(self, out_index, chs_per_bucket, blocks, out_shape):
       # reads output and returns torch tensor of shape out_shape
       output = np.zeros(out_shape)
       for i in range(blocks):
           h = self.feature_map.read_bucket(i + out_index)
           output[:, i * chs_per_bucket:(i + 1) * chs_per_bucket, :, :] = h
       return torch.from_numpy(output)

   def compute_model(self, image):
       # Start recording memory accesses
       self.params.map.storage.start_recording()
       self.feature_map.storage.start_recording()
       
       valid_blocks, chs = self.scheduler.get_block_data(self.net.get_layer_types())
       indices = (0, self.feature_blocks)

       self.write_input(image, valid_blocks[0], chs[0])
       out_index = self.compute_layers(indices, valid_blocks, chs)
       
       layer_data = self.net.get_layer_data()
       out_shape = layer_data[self.net.num_layers]
       chs_per_bucket = chs[self.net.num_layers]
       blocks = valid_blocks[self.net.num_layers]
       output = self.read_output(out_index, chs_per_bucket, blocks, out_shape)

       # Save memory accesses
       self.params.map.storage.save()
       self.feature_map.storage.save()

       self.feature_map.free_all_blocks()
       return output
