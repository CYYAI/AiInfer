#ifndef _YOLOV8_SEG_CUDA_HPP_
#define _YOLOV8_SEG_CUDA_HPP_
#include <memory>
#include "backend/tensorrt/trt_infer.hpp"
#include "common/model_info.hpp"
#include "common/utils.hpp"
#include "common/cv_cpp_utils.hpp"
#include "common/memory.hpp"
#include "pre_process/pre_process.cuh"
#include "post_process/post_process.cuh"

namespace tensorrt_infer
{
    namespace yolov8_cuda
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class YOLOv8Seg
        {
        public:
            YOLOv8Seg() = default;
            ~YOLOv8Seg();
            void initParameters(const std::string &engine_file, float score_thr = 0.5f,
                                float nms_thr = 0.45f); // 初始化参数
            void adjust_memory(int batch_size);         // 由于batch size是动态的，所以需要对gpu/cpu内存进行动态的申请

            // forward
            SegBoxArray forward(const Image &image);
            BatchSegBoxArray forwards(const std::vector<Image> &images);

            // 模型前后处理
            void preprocess_gpu(int ibatch, const Image &image,
                                shared_ptr<Memory<unsigned char>> preprocess_buffer, AffineMatrix &affine,
                                cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchSegBoxArray parser_box(int num_image);

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            // seg的scale
            float scale_to_predict_x = 0.0f;
            float scale_to_predict_y = 0.0f;

            // 仿射矩阵的声明
            std::vector<AffineMatrix> affine_matrixs;
            const uint8_t const_value = 114; // 图片resize补边时的值

            // 使用自定义的Memory类用来申请gpu/cpu内存
            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, bbox_predict_, output_boxarray_, segment_predict_;
            Memory<unsigned char> box_segment_predict_;

            // 使用cuda流进行操作
            cudaStream_t cu_stream;

            // time
            Timer timer;
        };
    }
}

#endif // _YOLOV8_SEG_CUDA_HPP_