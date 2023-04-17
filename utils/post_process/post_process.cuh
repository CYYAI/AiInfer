#ifndef _POST_PROCESS_HPP_CUDA_
#define _POST_PROCESS_HPP_CUDA_

#include <iostream>
#include <cuda_runtime.h>
#include "common/cuda_utils.hpp"

#define BLOCK_SIZE 32

namespace ai
{
    namespace postprocess
    {
        // 一般用于对yolov3/v5/v7/yolox的解析，如果你有其他任务模型的后处理需要cuda加速，也可写在这个地方
        // 默认一张图片最多的检测框是1024，可以通过传参或者直接修改默认参数改变
        void decode_detect_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                          float confidence_threshold, float *invert_affine_matrix,
                                          float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);
        // nms的cuda实现
        void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

        // yolov8后处理解析
        void decode_detect_yolov8_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, float *invert_affine_matrix,
                                                 float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, cudaStream_t stream);
    }
}
#endif // _POST_PROCESS_HPP_CUDA_