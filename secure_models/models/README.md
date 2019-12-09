# Memory Protection Scheme Models

This is the source code for the simulation of the memory protection models. The top-level script is pytorch.py. The CNN used for inferences as well as the functions for training and testing the model is specified in pytorch.py. The pytorch.py file also has the functions to initialize the memory protection models and run inference on an image using the models. The script predicts the same images without storage as well as with the baseline model, the random map model, and either the original or modifed ORAM model depending the parameter set in tree_manager.py. The time for inference on each of the models will also be saved. 

## CNN Specification

The simulation requires the CNN to be specified as a pytorch neural netowrk class with a features module. The supported layers in the features module include convolution, activation, and pooling layers. The model class must also define the function classify, that takes the output of the features module and return the output of the network (see the Alexnet class in pytorch.py).

## Memory Interface and Protection Schemes

The memory protection schemes use the encrypted block storage for RAM defined in PyORAM. The access schemes are implemented as a wrapper above the memory interface in storage_manager.py. The MemoryMap class implements the random map memory scheme. The MemoryTree class implements Path ORAM. The memory interface used by these protection schemes is the RamStorage class defined in storage_manager.py. The RamStorage class uses the encrypted block storage in PyORAM to read and write to memory.

## Parameter Storage

The Random Map and ORAM models use the ParameterStore class defined in parameter_store.py to store weights and biases. The parameters are all initially written to RAM memory using the MemoryMap class. Unlike feature maps, however, the mapping between parameter blocks and physical blocks is not freed every time the parameter block is read. The ParameterStore class saves the location in memory of the parameters necessary for the computation of a particular task in a layer. When a particular parameter block is needed, it can be indexed by the layer and task number. 

## Model Parser

The Random Map and ORAM models use the ModelParser class defined in model_parser.py. The ModelParser class parses a model and determines the number of layers and the types of layers. The ModelParser class also saves the function that needs to be computed for a layer. Given the layer number, an input feature block, and weights, the ModelParser class will compute the layer function.

## Baseline Model

The baseline model is represented by the SimpleModelManager class in simple_manager.py. Because the baseline model does not use any memory protection scheme, the SimpleModelManager directly uses the RAM memory interface defined in storage_manager.py. The predict_test_store function defined in pytorch.py simulates one computation of the baseline model. 

## Random Map Model

The random map model is represented by the RandomMapModel class in random_map.py. The Scheduler class in random_map.py is used to determine hyperparmeters of the model such as number of feature blocks per layer and the size of each feature block. The Scheduler class will also use the functions in scheduler.py to determine the tasks that must be computed for each layer, how the tasks will be parititioned into jobs, and the order of the jobs. The MAX_ON_CHIP_MEMORY and MAX_KERNELS parameters in random_map.py will determine how the schedule is set. MAX_ON_CHIP_MEMORY limits the number of feature blocks that will be read in at once for each job. MAX_KERNELS should be set to None to remove the limit on the number of tasks that can be computed for each job. When the RandomMapModel class is initialized, the number of feature blocks, the size of the feature blocks, and the schedule will be set. The weights will also be written to memory. To run inference on a new image, run the compute_model function of the RandomMapModel class.

## ORAM Model 

Both the original ORAM and modified ORAM models are implemented by the OramModel class in tree_manager.py. To simulate the original ORAM model, set the MODIFY parameter in tree_manager.py to False, and to simulate the modified ORAM model, set the MODIFY parameter in tree_manager.py to True. The TreeManager class in tree_manager.py determines the hyperparameters of the model - the number of feature blocks per layer and the size of each feature block. When the OramModel class is initialized, these hyperparameters will be set and the weights will be written to memory. To run inference on a new image, run the compute_model function of the OramModel class.
