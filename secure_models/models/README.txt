This version simplifies the original mess into two files because it omits all the functionality of the original project EXCEPT the baseline model with storage and no protection scheme.

There's no timing functionality, no training functionality (a pre-trained model is provided), no random map, neither ORAM nor modified ORAM, just the baseline storage model. However, the code is much nicer.

Run the top.py file in command line with Python 2 (Python 3 will not work because of how it treats int division):

$ cd <this_directory>

$ python2.7 top.py

NOTE: python2.7 for my prompt, could be different for you.

Ignore the warning message about the source code being edited.

------------------------------------------------------------------------------------------

How is the program structured?

1. Run the top file, along with the pre-trained model. 
2. Initialize the simulated storage (Encrypted Block Storage from PyORAM).
3. Retrieve weights and biases for every convolution layer (not ReLU or pooling layers!).
   a. This includes all the parameters of the weights/biases: 
      the shape, weight/bias data, the start block, and the number of blocks
   b. "Retrieve" is not the best term, since it first writes the weights and biases
      to the storage
   c. All this is part of the "write_model()" method
4. Now, read each input f-map, flatten (& pad if neces.) them, and store them in chunks.
   a. These chunks are "blocks," contiguously done without any shuffling
5. After every block of the input f-map data is stored, read the data.
6. Reshape the data and compute the convolutions --> output f-maps.
7. Store the output f-maps to the simulated memory.
   a. Again, just like before, writing back to storage is also done in chunks (by blocks)
8. Read the output f-maps (in chunks again) from the start block one last time.
9. Run a fully-connected layer to reduce the 256 layers to 10.
10. Out of the 10 possible outputs, choose the most likely option, the max value.