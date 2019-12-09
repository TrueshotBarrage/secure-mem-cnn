import torch
import torch.nn.functional as F

class ModelParser():
   def __init__(self, model, image):
      self.layer_data = dict()
      self.layer_data[0] = image.shape
      self.num_layers = 0
      self.layer_types = []
      self.conv_layers = dict() # store layer computation data for convolution layers
      self.software_layers = dict()

      x = image
      # determine layer shape data and types
      for (i, layer) in enumerate(model.features):
          self.num_layers += 1
          x = layer(x)
          self.layer_data[self.num_layers] = x.shape 
          params = list(layer.parameters())
          if (len(params) == 0):
             self.layer_types.append('SOFTWARE')
             self.software_layers[i] = layer
          else:
             self.layer_types.append('CONVOLUTION')
             self.conv_layers[i] = (layer.stride, layer.padding, layer.dilation)

   def get_layer_data(self):
      # returns a dictionary containing output shapes of layers
      # shape of input data is stored as layer 0
      return self.layer_data

   def get_layer_types(self):
      # returns list of layer types
      return self.layer_types

   def compute_task(self, layer_num, feature_block, params=None):
      # feature_block and params are torch tensors
      if (self.layer_types[layer_num] == 'CONVOLUTION'):
          stride, padding, dilation = self.conv_layers[layer_num]
          return F.conv2d(feature_block, params, stride = stride, padding = padding, dilation = dilation)
      else:
          layer = self.software_layers[layer_num]
          return layer(feature_block)
