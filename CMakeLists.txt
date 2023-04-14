cmake_minimum_required(VERSION 3.10.0)
# project(infer CXX C CUDA)
project(infer)
add_definitions(-std=c++11 -w)

# 要删除的设置,由于是本机是gcc/g++多版本,所以通过路径指定设置
set(CMAKE_C_COMPILER /home/yyds/softwares/gcc_g++_install/gcc)
set(CMAKE_CXX_COMPILER /home/yyds/softwares/gcc_g++_install/g++)

# 如果你是不同显卡，请设置为显卡对应的号码参考下面的链接，我这里是RTX 3090,对应的是sm_86：
# https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=sm_86;-G;-g;-O0;-w")

# 设置工作目录,里面会放测试图片和模型，生成的可执行文件也会在该目录下
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspaces)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/example/faceLibs/) # install时的存储路径

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(CUDA_DIR "/usr/local/cuda-11.1") # 设置cuda根目录，设置你自己的即可
set(TensorRT_ROOT "/home/yyds/softwares/TensorRT-7.2.3.4") # 设置tensorrt根目录，改为你自己的即可

include_directories(
    ${PROJECT_SOURCE_DIR}/application
    ${PROJECT_SOURCE_DIR}/utils
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${TensorRT_ROOT}/include
    ${TensorRT_ROOT}/samples/common # 导入这个主要是为了适应于trt多版本[v7.xx,v8.xx]的logger导入
)

link_directories(
    ${CUDA_DIR}/lib64
    ${TensorRT_ROOT}/lib
)

file(GLOB_RECURSE cuda_srcs
    ${PROJECT_SOURCE_DIR}/utils/*.cu)
cuda_add_library(code_cu SHARED ${cuda_srcs})

file(GLOB_RECURSE cpp_srcs
    ${PROJECT_SOURCE_DIR}/application/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/*.cpp
    ${TensorRT_ROOT}/samples/common/logger.cpp # 引用对应版本的logger.cpp，用来适应多版本
    ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
    )
add_library(code_cpp SHARED ${cpp_srcs})

add_executable(infer main.cpp ${cpp_srcs})
target_link_libraries(infer
    code_cpp
    nvinfer
    cuda
    cudart
    cudnn
    pthread
    ${OpenCV_LIBS}
    code_cu
)

# make install 时需要用到
install(TARGETS infer code_cu code_cpp
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/application/
        ${PROJECT_SOURCE_DIR}/utils/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")

# 通过make auto -j 来编译和运行程序
# add_custom_target(
#     auto
#     DEPENDS infer
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspaces
#     COMMAND ./infer -f weights/yolov8_face_mask_detect_0.951_int8_quantized_dynamic.engine -i res/test_demo.jpg
# )
