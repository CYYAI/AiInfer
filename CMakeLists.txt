cmake_minimum_required(VERSION 3.10.0)
project(infer)
add_definitions(-std=c++11 -w)

# 1. 设置工作目录,里面会放测试图片和模型，生成的可执行文件也会在该目录下
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspaces)
set(CMAKE_INSTALL_PREFIX ${EXECUTABLE_OUTPUT_PATH}/install/) # make install时的存储路径

# 2. 设置显卡算力，如果你是不同显卡，请设置为显卡对应的号码参考下面的链接，我这里是RTX 3090,对应的是sm_86：
# https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_86,code=sm_86;-G;-g;-O0;-w")

# 3. 寻找cuda和opencv库
find_package(CUDA REQUIRED) # 这个默认你本机已经安装
find_package(OpenCV REQUIRED) # 如果你没安装，sudo apt-get install libopencv-dev
find_package(Eigen3 REQUIRED) # 在使用追踪项目的时候要用到，其他项目直接注释掉即可

# 4. 设置tensorrt的主目录,支持tensorrt7.xx和tensorrt8.xx
# set(TensorRT_ROOT "/home/yyds/thrid_libs/TensorRT-7.2.3.4") # 设置tensorrt7.xx根目录，改为你自己的即可
set(TensorRT_ROOT "/home/yyds/thrid_libs/TensorRT-8.6.0.12") # 设置tensorrt8.xx根目录，改为你自己的即可

# 5. include所有要用到的hpp文件路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} # 追踪要用到

    # tensorrt
    ${TensorRT_ROOT}/include
    ${TensorRT_ROOT}/samples/common # 导入这个主要是为了适应于trt多版本[v7.xx,v8.xx]的logger导入

    # 项目里要用到的
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/application
)

# 6. link要用到的so库路径
# 补充：具体的cuda_lib库命名可以看 https://cmake.org/cmake/help/latest/module/FindCUDA.html
link_directories(
    # ${CUDA_LIBRARIES}
    # ${CUDA_cublas_LIBRARY}
    # ${CUDA_cudart_static_LIBRARY}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64

    # tensorrt
    ${TensorRT_ROOT}/lib
)


# 7. 将utils里写好的cu文件和cpp文件编译成so库，方便后面调用
file(GLOB_RECURSE cpp_cuda_srcs
    ${PROJECT_SOURCE_DIR}/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/application/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/*.cu
    ${TensorRT_ROOT}/samples/common/logger.cpp # 引用对应版本的logger.cpp，用来适应多版本
    ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
    ${TensorRT_ROOT}/samples/common/sampleUtils.cpp
)
cuda_add_library(utils_cu_cpp SHARED ${cpp_cuda_srcs})

# add_executable(infer mains/main_yolo_series_det.cpp)
# add_executable(infer mains/main_yolov8_det.cpp)
# add_executable(infer mains/main_yolov8_seg.cpp)
add_executable(infer mains/main_yolov8_pose.cpp)
# add_executable(infer mains/main_track_yolov8_det.cpp)
# add_executable(infer mains/main_rtdetr.cpp)
# 8. 链接要所有要用到的so库
target_link_libraries(infer
    utils_cu_cpp # 调用上面编译好的so库
    cuda
    cudart
    cudnn
    pthread
    ${OpenCV_LIBS}
    nvinfer 
    nvinfer_plugin
    # nvonnxparser    
)

# make install 时需要用到
install(TARGETS infer  utils_cu_cpp
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/utils/
        ${PROJECT_SOURCE_DIR}/application/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")

# 通过make auto -j 来编译和运行程序
# add_custom_target(
#     auto
#     DEPENDS infer
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspaces
#     COMMAND ./infer
# )
