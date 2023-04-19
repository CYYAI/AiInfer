cmake_minimum_required(VERSION 3.10.0)
project(infer)
add_definitions(-std=c++11 -w)

##########################开始设置要运行的目标平台######################################
set(TARGET_PLAFORM "openvino") # 设置你要运行的目标平台【tensorrt，openvino，ncnn，mnn】,暂时只支持这四种


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

# 设置要动态链接的动态库或者静态库
set(TARGET_LIN_LIBS
    cuda
    cudart
    cudnn
    pthread
    ${OpenCV_LIBS}
)

set(INCLUDE_PROJECT_DIRS
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/application
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# 具体的cuda_lib库命名可以看 https://cmake.org/cmake/help/latest/module/FindCUDA.html
set(LINK_PROJECT_DIRS
    # ${CUDA_LIBRARIES}
    # ${CUDA_cublas_LIBRARY}
    # ${CUDA_cudart_static_LIBRARY}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 设置要编译的所有cpp文件
set(COMPILE_CPPS
    ${PROJECT_SOURCE_DIR}/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/application/*.cpp
)

# 设置要install的库的hpp文件
set(INSTALL_INCLUDE_PATH
    ${PROJECT_SOURCE_DIR}/utils/
    ${PROJECT_SOURCE_DIR}/application/
)

if(${TARGET_PLAFORM} STREQUAL  "tensorrt")
    set(TensorRT_ROOT "/home/yyds/softwares/TensorRT-7.2.3.4") # 设置tensorrt7.xx/8.xx根目录，改为你自己的即可
    set(TRT_BACKEND  ${PROJECT_SOURCE_DIR}/backend/tensorrt)
    add_compile_definitions(TENSORRT) # 方便在源码中对要使用的backend进行判断
    
    # 添加要include的trt文件
    list(APPEND INCLUDE_PROJECT_DIRS 
        ${TensorRT_ROOT}/include
        ${TensorRT_ROOT}/samples/common # 导入这个主要是为了适应于trt多版本[v7.xx,v8.xx]的logger导入
        ${TRT_BACKEND} # 加载tensorrt推理库的hpp文件
    )
    # 添加要lib的trt动态或静态库
    list(APPEND LINK_PROJECT_DIRS
        ${TensorRT_ROOT}/lib
    )
    # 添加要编译的trt总要用到的cpp文件
    list(APPEND COMPILE_CPPS
        ${TensorRT_ROOT}/samples/common/logger.cpp # 引用对应版本的logger.cpp，用来适应多版本
        ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
        ${TRT_BACKEND}/*.cpp # 编译tensorrt推理库的cpp文件
    )

    # 添加要动态链接的trt库
    list(APPEND TARGET_LIN_LIBS
        nvinfer 
        nvinfer_plugin
        # nvonnxparser
    )
    
    # trt要追加install的库
    list(APPEND  INSTALL_INCLUDE_PATH 
        ${TRT_BACKEND}/
    )
elseif(${TARGET_PLAFORM} STREQUAL  "openvino")
    set(OPENVINO_ROOT "/home/yyds/thrid_libs/openvino/runtime")
    set(OPENVINO_BACKEND  ${PROJECT_SOURCE_DIR}/backend/openvino)
    add_compile_definitions(OPENVINO) # 方便在源码中对要使用的backend进行判断
    
    # 添加要include的openvino文件
    list(APPEND INCLUDE_PROJECT_DIRS 
        ${OPENVINO_ROOT}/include
        ${OPENVINO_BACKEND} # 加载openvino推理库的hpp文件
    )
    # 添加要lib的openvino动态或静态库
    list(APPEND LINK_PROJECT_DIRS
        ${OPENVINO_ROOT}/lib/intel64
    )
    # 添加要编译的openvino总要用到的cpp文件
    list(APPEND COMPILE_CPPS
        ${OPENVINO_BACKEND}/*.cpp
    )

    # 添加要动态链接的openvino库
    list(APPEND TARGET_LIN_LIBS
        openvino
    )
    
    # openvino要追加install的库
    list(APPEND  INSTALL_INCLUDE_PATH 
        ${OPENVINO_BACKEND}/
    )
endif()

include_directories(${INCLUDE_PROJECT_DIRS})
link_directories(${LINK_PROJECT_DIRS})

# 将utils里写好的cu文件编译成so库，方便后面调用
file(GLOB_RECURSE cuda_srcs
    ${PROJECT_SOURCE_DIR}/utils/*.cu
)
cuda_add_library(utils_cu SHARED ${cuda_srcs})

# 将上面设置的所有要编译的cpp文件，编译成so库
file(GLOB_RECURSE need_compile_cpps ${COMPILE_CPPS})
add_library(useful_cpps SHARED ${need_compile_cpps})

add_executable(infer main.cpp)
target_link_libraries(infer
    useful_cpps    
    ${TARGET_LIN_LIBS}
    utils_cu
    
)

# make install 时需要用到
install(TARGETS infer utils_cu useful_cpps
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
