import torch
import torch.nn.functional as F

import numpy
import random
import math
import time

from storage_manager import *
from model_parser import *
from parameter_store import *

import pdb

FEATURE_STORAGE_NAME = 'oram_features.bin'
PARAM_STORAGE_NAME = 'oram_parameters.bin'
MAX_ON_CHIP_MEMORY = 30 * 1024
STASH_OVERFLOW_BUFFER = 2
MAX_KERNELS = 100
SCALE_FACTOR = 2

MODIFY = True

class TreeManager():
   def __init__(self, num_layers, layer_data, block_size, elem_size = 4):
      # layer_data is a dictionary containing the shapes of the input feature
      # map for every layer. The shape of the output to the network is stored as
      # the data for layer num_layers. The shape of each layer should be (1, N, H, W)
      # layer_type is either 'CONVOLUTION' or 'SOFTWARE'
      self.num_layers = num_layers
      self.elem_size = elem_size

      self.ch_size = np.zeros(self.num_layers + 1, dtype = 'int32')
      self.num_chs = np.zeros(self.num_layers + 1, dtype = 'float32') 

      for i in range(self.num_layers + 1):
         self.ch_size[i] = layer_data[i][2] * layer_data[i][3]
         self.num_chs[i] = layer_data[i][1]

      self.block_size = block_size
      self.min_bucket_size = int(np.max(self.ch_size) * self.elem_size / self.block_size)
      self.max_bucket_size = int((MAX_ON_CHIP_MEMORY / block_size) / 2)

      if (self.max_bucket_size < self.min_bucket_size):
        print('[ERROR] CANNOT DETERMINE STORAGE PARAMETERS')

   def num_kernels(self, bucket_size):
      b = int(bucket_size / self.elem_size) # number of elements that can fit in bucket
      y = np.ceil(self.num_chs / (b / self.ch_size))
      return y.astype('int32')

   def stash_size(self, bucket_size, N):   
      tree_height = math.ceil(math.log(N, 2))
      return tree_height * bucket_size

   def set_storage_parameters(self):
      min_memory = 0
      min_kernels = 0
      min_bucket = 0
      for i in range(self.min_bucket_size, self.max_bucket_size + 1):
         bucket_size = i * self.block_size
         kernels = np.max(self.num_kernels(bucket_size))
         stash_size = self.stash_size(bucket_size, kernels)
         if (stash_size + STASH_OVERFLOW_BUFFER * bucket_size < MAX_ON_CHIP_MEMORY):
            max_blocks = int(MAX_ON_CHIP_MEMORY / bucket_size) * stash_size
            j = max_blocks / 2
            k = max_blocks - j
            memory = int(((kernels + j - 1) / j) * ((kernels + k - 1) / k) * bucket_size)
            if (min_memory == 0 or memory < min_memory):
               min_memory = memory
               min_kernels = kernels
               min_bucket = bucket_size

      if (min_memory == 0):
         print('[ERROR] NOT ENOUGH MEMORY')

      self.num_blocks = SCALE_FACTOR * min_kernels
      self.bucket_size = min_bucket # set in number of bytes
      
      return (self.num_blocks, self.bucket_size)

   def set_layer_schedule(self, layer_types):
      self.schedule = dict()
      conv_layer = -1
      for i in range(self.num_layers):
         append_layers = []
         if(layer_types[i] == 'CONVOLUTION'):
            self.schedule[i]  = append_layers
            conv_layer = i
         elif (layer_types[i] == 'SOFTWARE'):
            self.schedule[conv_layer].append(i)

   def get_layer_schedule(self, layer):
      # Returns software layers that must be appended to the current layer
      return self.schedule[layer]

   def get_block_data_pad(self, layer_types):
      # Returns valid block data and channels per bucket (padded to worst case to reduce size of parameter storage)
      blocks = np.where(self.num_chs < self.num_blocks, self.num_chs, self.num_blocks)
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

   def get_block_data(self, layer_types):
      # Returns valid block data and channels per bucket (not padded to worst case)
      valid_blocks = self.num_kernels(self.bucket_size)
      chs = np.ceil(self.num_chs / valid_blocks).astype('int32')
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

class OramModel():
    def __init__(self, model, image, block_size, block_count):
       # Parse model to get model data
       self.net = ModelParser(model, image)

       # Create manager and set storage parameters
       self.manager = TreeManager(self.net.num_layers, self.net.layer_data, block_size)
       print('[INFO] DETERMINING FEATURE STORAGE PARAMETERS')
       feature_blocks, feature_size  = self.manager.set_storage_parameters()
       print('[INFO] NUMBER OF FEATURE BLOCKS PER LAYER ' + str(feature_blocks))
       print('[INFO] SIZE OF FEATURE BLOCKS (NUMBER OF MEMORY BLOCKS)	' + str(feature_size / block_size))
       self.manager.set_layer_schedule(self.net.layer_types)      
       self.feature_blocks = feature_blocks
 
       # Write parameters
       valid_blocks, chs = self.manager.get_block_data_pad(self.net.layer_types)
       self.params = ParameterStore(model)
       num_blocks = self.params.get_block_count(model, self.net.layer_types, block_size, valid_blocks, chs)
       param_storage = RamStorage(PARAM_STORAGE_NAME, block_size, num_blocks)
       self.params.set_storage(param_storage)
       print('[INFO] SET UP PARAMETER STORAGE WITH ' + str(self.params.num_buckets) 
       + ' BUCKETS WITH ' + str(self.params.bucket_size) + ' BLOCKS')
       self.params.write_parameters(model, self.net.layer_types, valid_blocks, chs)
       print('[INFO] FINISHED WRITING WEIGHTS')

       # Create feature storage
       self.feature_tree = MemoryTree(FEATURE_STORAGE_NAME, block_size, feature_size / block_size, 2 * feature_blocks)
       print('[INFO] SET UP FEATURE STORAGE')

    def get_compute_table(self, valid_blocks):
       # NOT PADDED TO WORST CASE
       i, j = valid_blocks
       return np.zeros((i, j))

    def get_output_table(self, valid_blocks):
       i, j = valid_blocks
       return np.zeros(j)

    def get_weight_data(self, layer, tasks):
       # given set of tasks, returns dictionary of weights necessary for computation
       # dictionary is indexed by task
       weights = dict()
       for task in tasks:
          x, y = task
          weights[(x, y)] = torch.from_numpy(self.params.get_weights(layer, task))
       return weights

    def get_bias_data(self, layer, tasks, channels_bucket, output_table):
       # given set of tasks return the biases needed in a dictionary
       bias = dict() 
       for task in tasks:
          i, y = task
          if (y not in bias and output_table[y] == 0):
             _, out_ch = channels_bucket
             bias_data = self.params.get_bias(layer, y)
             num_ch = bias_data.shape[0]
             bias[y] = torch.from_numpy(bias_data.reshape(1, num_ch, 1, 1))
       return bias

    def compute_software_layers(self, layers, feature_data, feature_blocks, indices):
       # given list of feature_blocks that are done, will compute layers
       for block in feature_blocks:
          index = block + indices[1]
          for layer in layers:
             feature_data[index] = self.net.compute_task(layer, feature_data[index])

    def compute_tasks(self, layer, tasks, feature_data, weights, bias, valid, indices, compute_table, output_table):
       # given set of tasks, compute the tasks
       for task in tasks:
          x, y = task
          compute_table[x, y] = 1
          i = x + indices[0]
          j = y + indices[1]
          if (x < valid[0] and y < valid[1]):
             if (output_table[y] == 0):
                feature_data[j] = self.net.compute_task(layer, feature_data[i], weights[task])
                # add bias
                feature_data[j] += bias[y]
                output_table[y] = 1
             else:
                out = self.net.compute_task(layer, feature_data[i], weights[task])
                feature_data[j] += out
                output_table[y] += 1
       # check for finished feature map blocks
       blocks_finished = []
       for y in range(valid[1]):
          if (output_table[y] == valid[0]):
             blocks_finished.append(y)
             output_table[y] += 1 
       return blocks_finished
 
    def add_tasks(self, feature_data, compute_table, indices, valid_blocks, task = None):
       # given feature data, returns all possible tasks that can be computed
       tasks = set()
       kernels = 0
       if (task is not None):
          tasks.add(task)
          kernels += 1

       tasks_left = np.where(compute_table == 0)
       num_tasks_left = (tasks_left[0]).size
       for n in range(num_tasks_left):
          x = tasks_left[0][n]
          y = tasks_left[1][n]
          i = x + indices[0]
          j = y + indices[1]
          if (i in feature_data and j in feature_data):
             if (x < valid_blocks[0] and y < valid_blocks[1]):
                if ((x, y) not in tasks):
                   tasks.add((x, y))
                   kernels += 1
                   if (MAX_KERNELS is not None and kernels >= MAX_KERNELS):
                      break
       return tasks
       
    def complete_tasks(self, layer, software_layers, tasks, feature_data, indices, 
    valid_blocks, chs, compute_table, output_table):
       weights = self.get_weight_data(layer, tasks)
       bias = self.get_bias_data(layer, tasks, chs, output_table)
       # convert feature data to torch tensor
       for i in feature_data:
          feature_data[i] = torch.from_numpy(feature_data[i])

       blocks_finished = self.compute_tasks(layer, tasks, feature_data, weights, 
       bias, valid_blocks, indices, compute_table, output_table)
       self.compute_software_layers(software_layers, feature_data, blocks_finished, indices)
       # convert feature data back to numpy
       for i in feature_data:
           feature_data[i] = feature_data[i].detach().numpy()

    def compute_job(self, layer, software_layers, indices, valid_blocks, chs, compute_table, output_table):
       tasks_left = np.where(compute_table == 0)
       inputs = tasks_left[0]
       outputs = tasks_left[1]
       task_num = random.randint(0, inputs.size - 1)
      
       x = inputs[task_num]
       y = outputs[task_num]
       i  = x + indices[0]
       j = y + indices[1]
  
       # read in stash for input
       feature_data = self.feature_tree.read_bucket(i)
       # save input data
       overflow_index = i
       overflow_data = feature_data[i]
       # compute all tasks that can be computed
       tasks = self.add_tasks(feature_data, compute_table, indices, valid_blocks)
       self.complete_tasks(layer, software_layers, tasks, feature_data, indices, valid_blocks, chs, compute_table, output_table)
       self.feature_tree.write_path(i, feature_data)

       # check if task to compute is complete
       if (compute_table[x, y] == 0):
          # task has not been completed
          feature_data = self.feature_tree.read_bucket(j)
          feature_data[overflow_index] = overflow_data
          tasks = self.add_tasks(feature_data, compute_table, indices, valid_blocks, task =(x, y))
          self.complete_tasks(layer, software_layers, tasks, feature_data, indices, valid_blocks, chs, compute_table, output_table)
          self.feature_tree.write_path(j, feature_data)             
       
    def compute_job_original(self, layer, software_layers, indices, valid_blocks, chs, compute_table, output_table):
       tasks_left = np.where(compute_table == 0)
       inputs = tasks_left[0]
       outputs = tasks_left[1]
       task_num = random.randint(0, inputs.size - 1)

       x = inputs[task_num]
       y = outputs[task_num]
       i = x + indices[0]
       j = y + indices[1]

       # read in stash for input
       feature_data = self.feature_tree.read_bucket(i)
       # save input data
       overflow_index = i
       overflow_data = feature_data[i]

       if (j not in feature_data):
          # read in output feature data
          feature_data = self.feature_tree.read_bucket(j)
          feature_data[overflow_index] = overflow_data
       
       tasks = {(x, y)}
       self.complete_tasks(layer, software_layers, tasks, feature_data, indices, valid_blocks, chs, compute_table, output_table)
       self.feature_tree.write_path(i, feature_data)      
 
    def compute_layer(self, layer_num, indices, valid_blocks, chs):
       software_layers = self.manager.get_layer_schedule(layer_num)
       compute_table = self.get_compute_table(valid_blocks)       
       output_table = self.get_output_table(valid_blocks)
       
       i, j = valid_blocks 
       while(np.sum(compute_table) < i * j):     
          if (MODIFY):
             self.compute_job(layer_num, software_layers, indices, valid_blocks, chs, compute_table, output_table)
          else:
             self.compute_job_original(layer_num, software_layers, indices, valid_blocks, chs, compute_table, output_table)
       return (indices[1], indices[0])

    def compute_layers(self, indices, valid_blocks, chs):
       layer_types = self.net.get_layer_types()

       for i in range(self.net.num_layers):
          layer_type = layer_types[i]
          if (layer_type == 'CONVOLUTION'):
             valid = (valid_blocks[i], valid_blocks[i + 1])
             channels_per_bucket = (chs[i], chs[i + 1])
             indices = self.compute_layer(i, indices, valid, channels_per_bucket)

       return indices[0]
       
    def write_input(self, image, valid_blocks, chs_bucket):
       start_time = time.time()      
       # Writes one channel to one bucket
       for i in range(valid_blocks):
          data = image[:, i * chs_bucket:(i + 1) * chs_bucket, :, :]
          self.feature_tree.write_bucket(i, data.detach().numpy())

       # Write zeros to all other blocks in oram tree
       for i in range(valid_blocks, self.feature_tree.num_buckets):
          data = np.zeros(1, dtype='float32')
          self.feature_tree.write_bucket(i, data)
       end_time = time.time()
       #print('Time to write image:	' + str(end_time - start_time))

    def read_output(self, out_index, chs_per_bucket, blocks, out_shape):
      # reads output and returns torch tensor of shape out_shape
      output = np.zeros(out_shape)
      for i in range(blocks):
         stash = self.feature_tree.read_bucket(i + out_index)
         output[:, i * chs_per_bucket:(i + 1) * chs_per_bucket, :, :] = stash[i + out_index]
      return torch.from_numpy(output)

    def compute_model(self, image):
      # Start recording memory accesses
      self.params.map.storage.start_recording()
      self.feature_tree.storage.start_recording()

      valid_blocks, chs = self.manager.get_block_data_pad(self.net.get_layer_types())
      indices = (0, self.feature_blocks)

      print('[INFO] WRITING INPUT')
      self.write_input(image, valid_blocks[0], chs[0])
      out_index = self.compute_layers(indices, valid_blocks, chs)

      layer_data = self.net.get_layer_data()
      out_shape = layer_data[self.net.num_layers]
      chs_per_bucket = chs[self.net.num_layers]
      blocks = valid_blocks[self.net.num_layers]
      output = self.read_output(out_index, chs_per_bucket, blocks, out_shape)

      # Save memory accesses
      self.params.map.storage.save()
      self.feature_tree.storage.save()

      self.feature_tree.reset()
      return output     
  
