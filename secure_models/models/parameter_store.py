import numpy as np
import torch 

from storage_manager import *

class ParameterStore():
   def __init__(self, model):
      # weights indexed by (layer, task)
      self.weights = dict()
      # bias indexed by layer
      self.bias = dict()

   def num_blocks_weights(self, layer, block_size, in_ch, out_ch):
      num_bytes = layer.weight[:out_ch, :in_ch, :, :].detach().numpy().nbytes
      num_blocks = (num_bytes + block_size - 1) / block_size
      return num_blocks

   def num_blocks_bias(self, layer, block_size, out_ch):
      num_bytes = layer.bias[:out_ch].detach().numpy().nbytes
      num_blocks = (num_bytes + block_size - 1) / block_size
      return num_blocks
 
   def bucket_count(self, in_blocks, out_blocks):
      return (in_blocks + 1) * out_blocks

   def get_block_count(self, model, layer_types, block_size, valid_blocks, ch_per_block, pad = False):
      # layer_types is list 
      max_bucket_size = 0 # in number of blocks
      num_buckets = np.zeros(len(valid_blocks))
      num_layers = 0
      for (i, layer) in enumerate(model.features):
         if (layer_types[i] == 'CONVOLUTION'):
            num_layers += 1   
            # determine necessary bucket size
            bucket_size_weights = self.num_blocks_weights(layer, block_size, ch_per_block[i], ch_per_block[i + 1])
            bucket_size_bias = self.num_blocks_bias(layer, block_size, ch_per_block[i + 1])
            bucket_size = max(bucket_size_weights, bucket_size_bias)
            max_bucket_size = max(max_bucket_size, bucket_size)
            # determine number of buckets needed
            n = self.bucket_count(valid_blocks[i], valid_blocks[i + 1])
            num_buckets[i] = n      
      self.bucket_size = max_bucket_size
      if (pad):
         self.num_buckets = int(np.max(num_buckets)) * num_layers
      else:
         self.num_buckets = int(np.sum(num_buckets)) 
      return self.num_buckets * self.bucket_size

   def set_storage(self, storage):
      self.map = MemoryMap(storage, 0, self.bucket_size, self.num_buckets) 

   def write_layer_weights(self, l, weights, in_blocks, out_blocks, in_ch, out_ch, bucket):
      for i in range(in_blocks):
         for j in range(out_blocks):
            h = weights[j * out_ch:(j + 1) * out_ch, i * in_ch:(i + 1) * in_ch, :, :]
            self.map.write_bucket(bucket, h)
            self.weights[(l, (i, j))] = bucket
            bucket += 1
      return bucket

   def write_layer_bias(self, l, bias, bucket, out_blocks, out_ch):
      for i in range(out_blocks):
          h = bias[i * out_ch:(i + 1) * out_ch]
          self.map.write_bucket(bucket, h)
          self.bias[(l, i)] = bucket
          bucket += 1
      return bucket

   def write_dummy_weights(self, l, x, y, num_blocks, bucket):
      for i in range(x, num_blocks):
         for j in range(num_blocks):
            h = np.zeros(1, dtype='int32')
            self.map.write_bucket(bucket, h)
            self.weights[(l, (i, j))] = bucket
            bucket += 1
      for j in range(y, num_blocks):
         for i in range(x, num_blocks):
            h = np.zeros(1, dtype='int32')
            self.map.write_bucket(bucket, h)
            self.weights[(l, (i, j))] = bucket
            bucket += 1
      return bucket
 
   def write_parameters(self, model, layer_types, valid_blocks, ch_per_block, num_blocks = None):
      param_bucket = 0
      for (i, layer) in enumerate(model.features):
         if (layer_types[i] == 'CONVOLUTION'):
            in_blocks = valid_blocks[i]
            out_blocks = valid_blocks[i + 1]
            param_bucket = self.write_layer_weights(i, layer.weight.detach().numpy(), in_blocks, out_blocks, ch_per_block[i], ch_per_block[i + 1], param_bucket)
            if (num_blocks is not None):
               param_bucket = self.write_dummy_weights(i, in_blocks, out_blocks, num_blocks, param_bucket)
            param_bucket = self.write_layer_bias(i, layer.bias.detach().numpy(), param_bucket, valid_blocks[i + 1], ch_per_block[i + 1])

   def get_weights(self, layer, task):
      bucket = self.weights[(layer, task)]
      return self.map.read_bucket(bucket, free=False)

   def get_bias(self, layer, output_block):
      bucket = self.bias[(layer, output_block)]
      return self.map.read_bucket(bucket, free=False)
