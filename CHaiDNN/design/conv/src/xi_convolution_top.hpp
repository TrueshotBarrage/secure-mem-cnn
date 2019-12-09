#include "../include/xi_conv_config.h"
#include "../include/xi_conv_desc.h"

#if XI_DP_ENABLE
#include "../include/dsp_builtins.h"
#endif

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
);

