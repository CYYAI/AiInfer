cmake_minimum_required(VERSION 3.10.0)
project(infer)
add_definitions(-std=c++11 -w)

# 设置工作目录,里面会放测试图片和模型，生成的可执行文件也会在该目录下
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspaces)
set(CMAKE_INSTALL_PREFIX ${EXECUTABLE_OUTPUT_PATH}/install/) # make install时的存储路径

# 默认是在gpu上运行的，如果你仅仅想单独在cpu上运行，需要修改cmakelist和源码
# 如果你是不同显卡，请设置为显卡对应的号码参考下面的链接，我这里是RTX 3090,对应的是sm_86：
# https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=sm_86;-G;-g;-O0;-w")

# 寻找cuda和opencv库
find_package(CUDA REQUIRED) # 这个默认你本机已经安装
find_package(OpenCV REQUIRED) # 如果你没安装，sudo apt-get install libopencv-dev

include_directories(
    ${PROJECT_SOURCE_DIR}/utils
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
# 具体的cuda_lib库命名可以看 https://cmake.org/cmake/help/latest/module/FindCUDA.html
link_directories(
    # ${CUDA_LIBRARIES}
    # ${CUDA_cublas_LIBRARY}
    # ${CUDA_cudart_static_LIBRARY}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 将utils里写好的cu文件编译成so库，方便后面调用
file(GLOB_RECURSE cuda_srcs
    ${PROJECT_SOURCE_DIR}/utils/*.cu
    ${PROJECT_SOURCE_DIR}/utils/*.cpp
)
cuda_add_library(utils_cpp_cu SHARED ${cuda_srcs})

set(THRID_LINK_LIBS
    cuda
    cudart
    cudnn
    pthread
    ${OpenCV_LIBS}
    utils_cpp_cu
)

# 设置要install的库文件
set(INSTALL_LIBS utils_cpp_cu)
set(INSTALL_INCLUDE_PATH
    ${PROJECT_SOURCE_DIR}/utils/
)
##########################开始设置要运行的目标平台######################################
set(TARGET_PLAFORM "tensorrt") # 设置你要运行的目标平台【tensorrt，openvino，ncnn，mnn】,暂时只支持这四种

if(${TARGET_PLAFORM} STREQUAL  "tensorrt")
    set(TensorRT_ROOT "/home/yyds/softwares/TensorRT-7.2.3.4") # 设置tensorrt根目录，改为你自己的即可
    include_directories(
        ${TensorRT_ROOT}/include
        ${TensorRT_ROOT}/samples/common # 导入这个主要是为了适应于trt多版本[v7.xx,v8.xx]的logger导入
        ${PROJECT_SOURCE_DIR}/backend/tensorrt # 加载tensorrt推理库的hpp文件
        ${PROJECT_SOURCE_DIR}/application/tensorrt_demo
    )
    link_directories(
        ${TensorRT_ROOT}/lib
    )
    file(GLOB_RECURSE tensorrt_cpps
        ${TensorRT_ROOT}/samples/common/logger.cpp # 引用对应版本的logger.cpp，用来适应多版本
        ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
        ${PROJECT_SOURCE_DIR}/backend/tensorrt/*.cpp # 编译tensorrt推理库的cpp文件
        ${PROJECT_SOURCE_DIR}/application/tensorrt_demo/*.cpp
    )
    add_library(trt_cpp SHARED ${tensorrt_cpps})
    add_executable(infer main/main_trt.cpp ${trt_cpp} ${cpp_srcs})
    target_link_libraries(infer trt_cpp ${THRID_LINK_LIBS} nvinfer)

    # trt要追加install的库
    list(APPEND  INSTALL_LIBS trt_cpp)
    list(APPEND  INSTALL_INCLUDE_PATH 
        ${PROJECT_SOURCE_DIR}/backend/tensorrt/
        ${PROJECT_SOURCE_DIR}/application/tensorrt_demo/
    )
endif()


# make install 时需要用到
install(TARGETS infer ${INSTALL_LIBS}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY
        ${INSTALL_INCLUDE_PATH}
        DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")

# 通过make auto -j 来编译和运行程序
# add_custom_target(
#     auto
#     DEPENDS infer
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspaces
#     COMMAND ./infer
# )
