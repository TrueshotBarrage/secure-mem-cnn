/*----------------------------------------------------
Copyright 2017 Xilinx, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
----------------------------------------------------*/

#include "/media/hd/Vivado/2018.3/include/gmp.h"
#include "/media/hd/Vivado/2018.3/include/mpfr.h"

#include "../include/xi_conv_config.h"
#include "../include/xi_conv_desc.h"
typedef ap_int<27> dsp27_t;
#if XI_DP_ENABLE
#include "../include/dsp_builtins.h"
#endif
#include "xi_convolution.hpp"
#include "xi_convolution_top.hpp"

void XiConvolutionTop(
				gmem_weighttype *weights1,
				gmem_weighttype *weights2,
#if (XI_KER_PROC==16 || (XI_WTS_PORT_64BIT_EN==1 && XI_KER_PROC==8))
				gmem_weighttype *weights3,
				gmem_weighttype *weights4,
#endif
				gmem_outputtype *output1,
#if !XI_SINGLE_IO_PORT_EN
				gmem_outputtype *output2,
#endif
				gmem_inputtype_layerx *input_other1,
#if !XI_SINGLE_IO_PORT_EN
				gmem_inputtype_layerx *input_other2,
#endif
				gmem_inputtype_layer1 *input_1st,
				gmem_biastype *bias,
#if !XI_DISABLE_BN
				gmem_inputtype_layer1 *inp_norm_2,
				gmem_inputtype_layer1 *inp_norm_3,
#endif
				gmem_inputtype_layer1 *istg_out1,
#if !XI_SINGLE_IO_PORT_EN
				gmem_inputtype_layer1 *istg_out2,
#endif
				int *scalar_conv_args
) {
	bool ap_clk_div2 = 0;


#pragma HLS interface ap_bus port=weights1 		depth=500
#pragma HLS interface ap_bus port=weights2 		depth=500
#if (XI_KER_PROC==16 || (XI_WTS_PORT_64BIT_EN==1 && XI_KER_PROC==8) )
#pragma HLS interface ap_bus port=weights3 		depth=500
#pragma HLS interface ap_bus port=weights4 		depth=500
#endif
#pragma HLS interface ap_bus port=output1 		depth=1200
#pragma HLS interface ap_bus port=output2 		depth=1200
#pragma HLS interface ap_bus port=input_other1 	        depth=1200
#pragma HLS interface ap_bus port=input_other2 		depth=1200
#pragma HLS interface ap_bus port=input_1st 		depth=230400
#pragma HLS interface ap_bus port=bias 			depth=32
#pragma HLS interface ap_bus port=inp_norm_2 		depth=1200
#pragma HLS interface ap_bus port=inp_norm_3 		depth=1200
#pragma HLS interface ap_bus port=istg_out1 		depth=1200
#pragma HLS interface ap_bus port=istg_out2 		depth=1200
#pragma HLS interface ap_bus port=scalar_conv_args 	depth=128


/*
#pragma HLS interface s_axilite port=weights1 bundle=control
#pragma HLS interface s_axilite port=weights2 bundle=control
#if (XI_KER_PROC==16 || (XI_WTS_PORT_64BIT_EN==1 && XI_KER_PROC==8) )
#pragma HLS interface s_axilite port=weights3 bundle=control
#pragma HLS interface s_axilite port=weights4 bundle=control
#endif
#pragma HLS interface s_axilite port=output1 bundle=control
#pragma HLS interface s_axilite port=output2 bundle=control
#pragma HLS interface s_axilite port=input_other1 bundle=control
#pragma HLS interface s_axilite port=input_other2 bundle=control
#pragma HLS interface s_axilite port=input_1st bundle=control
#pragma HLS interface s_axilite port=bias bundle=control
#pragma HLS interface s_axilite port=scalar_conv_args bundle=control
#pragma HLS interface s_axilite port=inp_norm_2 bundle=control
#pragma HLS interface s_axilite port=inp_norm_3 bundle=control
#pragma HLS interface s_axilite port=istg_out1 bundle=control
#pragma HLS interface s_axilite port=istg_out2 bundle=control 

#pragma HLS interface s_axilite port=return bundle=control*/


	Convolution<XI_IN_W,XI_IN_H,XI_OUT_W,XI_OUT_H,XI_NUM_KERNELS,XI_FILTER_SIZE,XI_CONV_STRIDE,XI_POOL_STRIDE,XI_POOL_SIZE,XI_INPUT_PLANES,XI_NKPF>
	(weights1,weights2,
#if (XI_KER_PROC==16 || (XI_WTS_PORT_64BIT_EN==1 && XI_KER_PROC==8) )
			weights3,weights4,
#endif
			output1,
#if !XI_SINGLE_IO_PORT_EN
			output2,
#endif
			input_other1,
#if !XI_SINGLE_IO_PORT_EN
			input_other2,
#endif
			input_1st,bias,
#if !XI_DISABLE_BN
			inp_norm_2,inp_norm_3,
#endif
			istg_out1,
#if !XI_SINGLE_IO_PORT_EN
			istg_out2,
#endif
			scalar_conv_args,ap_clk_div2);
}

