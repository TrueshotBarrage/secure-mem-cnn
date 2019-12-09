# Secure CNN Computation Models

The CNN computation models simulate the computation of a CNN on an accelerator with different memory protection schemes. The following models are implemented here:

  1. Baseline model
  2. Original ORAM model
  3. Modified ORAM model
  4. Random Map model

These models are computed as described in the report. All of the source code for the implementation of the simulation is in the models directory. 

## Setup

### Install RAM

The software models use [PyORAM](https://github.com/ghackebeil/PyORAM), a python implementation of encrypted RAM storage and Oblivious RAM algorithms. The RAM directory contains all of the necessary source code for the memory interface. To install PyORAM, execute the following command from the RAM directory:

```shell
python setup.py install
```

### Train the model

The CNN used for the modeling is specified in models/pytorch.py. To train the model on the CIFAR-10 dataset, run the following command from the models directory:

```shell
python pytorch.py train [n]
```

Here, n is a parameter that specifies the number of images to predict after training the model. Once the model is trained, it will be saved to the model sub-directory.

## How to Run

Once a model is trained, to predict images again without training, run the following command from the models directory:

```shell
python pytorch.py load [n]
```

Again, n specifes the number of images that will be predicted. Each image is predicted using four different memory models:

  1. No memory storage. This predicts the image without writing or reading any data from memory.
  2. Baseline Model. This model predicts the image by writing and reading the feature map data and parameters from memory without any protection scheme.
  3. Random Memory Model. This model predicts the image using the random map algorithm.
  4. Tree Model. This model predicts the image using either the original ORAM memory protection scheme or the modified ORAM algorithm, depending on the parameter set in the models/tree_manager.py.
  
## Getting the Memory Access Pattern
 
Once the predictions are run, the memory accesses are saved as .npz files. To plot the memory access patterns and obtain the number of memory accesses for each model, run the parse.py script. The plots will be generated and saved in the plots sub-directory. The name of the npz file must be passed in as a command line argument to the script. 

The name of the npz file for each model is specified in the top-level file for the model. For the baseline model, the memory access trace is saved as heap.npz. For the random map memory model, the memory accces trace for the feature maps is saved as features.npz, and the memory access trace for the parameters is saved as parameters.npz. For the tree model, the memory access trace for the features map is saved as oram_features.npz, and the trace for the parameters is saved as oram_parameters.npz. 

For example, to get the memory trace for the feature maps with the random map model, execute the following command:

```shell
python parse.py features
```

To get the memory trace for the feature maps and parameters with the random map model, execute the following command:
```shell
python parse.py features parameters
```
