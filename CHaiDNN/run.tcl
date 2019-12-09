set part_name "xczu9eg-ffvb1156-2-e"
set project_name "zcu102"

open_project -reset $project_name

set_top DnnWrapper

add_files -tb software/bufmgmt/xi_buf_mgmt.cpp -cflags "-std=c++0x -I./software/bufmgmt"

add_files -tb software/checkers/checkers.cpp -cflags "-std=c++0x -I./software/checkers"

add_files -tb software/common/xi_kernels.cpp -cflags "-std=c++0x -I./software/common"

add_files -tb software/common/kernelinfo_class.cpp -cflags "-std=c++0x -I./software/common -I/software/scheduler"

add_files -tb software/custom/custom_class.cpp -cflags "-std=c++0x -I./software/custom"

add_files -tb software/imageread/xi_input_image.cpp -cflags "-std=c++0x -I./software/imageread"

add_files -tb software/include/xchange_structs.cpp -cflags "-std=c++0x -I./software/include"

add_files -tb software/init/xi_init.cpp -cflags "-std=c++0x -I./software/init"

add_files -tb software/interface/xi_interface.cpp -cflags "-std=c++0x -I./software/interface -I./software/xtract -I./software/scheduler -I./software/bufmgmt -I/media/hd/Vivado/2018.3/data/simmodels/xsim/2018.3/lnx64/6.2.0/ext/protobuf/include"

add_files -tb software/interface/xi_readwrite_util.cpp -cflags "-std=c++0x -I./software/interface"

add_files -tb software/scheduler/xi_perf_eval.cpp -cflags "-std=c++0x -I./software/scheduler -I./software/common"

add_files -tb software/scheduler/xi_scheduler.cpp -cflags "-std=c++0x -I./software/scheduler"

add_files -tb software/scheduler/xi_thread_routines.cpp -cflags "-std=c++0x -I./software/scheduler -I./software/include -I./software/common"

add_files -tb software/scheduler/xi_utils.cpp -cflags "-std=c++0x -I./software/scheduler"

set files [glob -directory "software/swkernels" "*.cpp"]
foreach file $files {
add_files -tb $file -cflags "-std=c++0x -I./software/include -I./software/swkernels"
}

set files [glob -directory "software/xtract" "*.cpp" "*.proto"]
foreach file $files {
add_files -tb $file -cflags "-std=c++0x -I/media/hd/Vivado/2018.3/data/simmodels/xsim/2018.3/lnx64/6.2.0/ext/protobuf/include"
}

add_files -tb software/xtract/caffe.pb.cc -cflags "-std=c++11 -I/media/hd/Vivado/2018.3/data/simmodels/xsim/2018.3/lnx64/6.2.0/ext/protobuf/include"

add_files design/wrapper/dnn_wrapper.cpp -cflags "-std=c++0x -D__HW__ -g"

add_files design/conv/src/xi_convolution_top.cpp -cflags "-std=c++0x -D__HW__ -g"

add_files design/pool/src/pooling_layer_dp_2xio_top.cpp -cflags "-std=c++0x -D__HW__ -g"

add_files design/deconv/src/xi_deconv_top.cpp -cflags "-std=c++0x -D__HW__ -g"

add_files -tb "software/example/lenet_ex.cpp" -cflags "-std=c++0x -I./software/interface -I./software/checkers"

open_solution "solution1"
set_part $part_name

create_clock -period 11

csim_design -ldflags "-L/media/hd/Vivado/2018.3/data/simmodels/xsim/2018.3/lnx64/6.2.0/ext/protobuf -lcblas -lprotobuf"

csynth_design 

cosim_design -ldflags "-L/media/hd/Vivado/2018.3/data/simmodels/xsim/2018.3/lnx64/6.2.0/ext/protobuf -lcblas -lprotobuf" -trace_level port
