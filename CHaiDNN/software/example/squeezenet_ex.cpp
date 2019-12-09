#include <stdio.h>
#include <string.h>
#include <stdlib.h>


#undef __ARM_NEON__
#undef __ARM_NEON
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define __ARM_NEON__
#define __ARM_NEON

#include <iostream>
using namespace std;
using namespace cv;

#include "../interface/xi_interface.hpp"
#include "../interface/xi_readwrite_util.hpp"
#include "../checkers/checkers.hpp"

int main(int argc, char **arcv) {
    io_layer_info io_layer_info_ptr1;
    io_layer_info io_layer_info_ptr2;

    char *dirpath    = "/home/mpa46/CHaiDNN/software/example/SqueezeNet/SqueezeNet_v1.0"; /* Path to the network model directory */
    char *prototxt   = "deploy.prototxt";       /* Prototxt file name residing in network model directory */
    char *caffemodel = "squeezenet_v1.0.caffemodel";  /* caffemodel file name residing in network model directory */
    
    char *img_path1  = "/home/mpa46/CHaiDNN/software/example/image_3.jpg";
    char *img_path2  = "/home/mpa46/CHaiDNN/software/example/image_3.jpg";

    string start_layer = "";
    string end_layer = "";
   
    int numImg_to_process = 1;
    bool is_first_layer_1 = 1;
    //bool is_first_layer_2 = 0;
    
    void *chai_handle1 = xiInit(dirpath, prototxt, caffemodel, &io_layer_info_ptr1, 
				numImg_to_process, is_first_layer_1, start_layer, end_layer);
 
    //void *chai_handle2 = xiInit(dirpath, prototxt, caffemodel, &io_layer_info_ptr2,
    //				numImg_to_process, is_first_layer_2, start_layer, end_layer);


    int inp_mode = 0;
    float *mean_ptr = (float*)malloc(sizeof(float));
    *mean_ptr = 0;

    int resize_h = 227;
    int resize_w = 227;

    vector<void *> normalizeInput;
    for (int batch_id = 0; batch_id < numImg_to_process; batch_id++) {
	  void *ptr = (int8_t *)malloc(resize_h*resize_w*sizeof(int8_t));
     	normalizeInput.push_back(ptr);
    }
    
    int status = inputNormalization(normalizeInput, resize_h, resize_w, img_path1, img_path2, 
    				inp_mode, mean_ptr, NULL, numImg_to_process, io_layer_info_ptr1);

    if (status == -1) {
	      fprintf(stderr, "[ERROR] IMAGE READ FAIL\n");
        free(mean_ptr);
	  return -1;
    } else {
        fprintf(stderr, "[INFO] IMAGE READ SUCCESS\n");
    }    
    
    int in_size = io_layer_info_ptr1.inlayer_sizebytes;
    //# Create input/output Buffers
    vector<void *> input;
    void *ptr;
    for(int i = 0; i < io_layer_info_ptr1.num_in_bufs; i++) {
     if(io_layer_info_ptr1.inlayer_exectype.compare("hardware") == 0)
       ptr = malloc(in_size);
     else
       ptr = malloc(in_size);
       input.push_back(ptr);
    }
    
    int out_size = io_layer_info_ptr1.outlayer_sizebytes;
    vector<void *> output;

    for(int i = 0; i < io_layer_info_ptr1.num_out_bufs; i++){
      if(io_layer_info_ptr1.outlayer_exectype.compare("hardware") == 0)
        ptr = malloc(out_size);
      else
        ptr = malloc(out_size);
        output.push_back(ptr);
    }
    
    xiInputRead(normalizeInput, input, numImg_to_process, io_layer_info_ptr1);
    xiExec(chai_handle1, input, output);

    fprintf(stderr, "[INFO] FINISHED EXECUTION\n");
    fflush(stderr);    

    int unpack_out_size = io_layer_info_ptr1.outlayer_sizebytes;

    //# Create memory for unpack output data
    vector<void *> unpack_output;
    for(int batch_id = 0; batch_id < numImg_to_process; batch_id++)
    {
      void *ptr = malloc(unpack_out_size);
      unpack_output.push_back(ptr);
    }
    //# Loading required params for unpack function
    kernel_type_e out_kerType = io_layer_info_ptr1.out_kerType;
    int out_layer_size = io_layer_info_ptr1.out_size;
    //# unpacks the output data
    xiUnpackOutput(output, unpack_output, out_kerType, out_layer_size, numImg_to_process);
    //# Write the output data to txt file
    outputWrite(dirpath, img_path1, unpack_output, numImg_to_process, io_layer_info_ptr1, 0);
    
    xiRelease(chai_handle1); //# Release before exiting application
    //xiRelease(chai_handle2);
    free(mean_ptr);
}

