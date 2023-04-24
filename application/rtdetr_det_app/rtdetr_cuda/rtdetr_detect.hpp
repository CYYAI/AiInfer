#ifndef _RTDETR_DETECT_CUDA_HPP_
#define _RTDETR_DETECT_CUDA_HPP_
#include <memory>
#include <algorithm>
#include "backend/tensorrt/trt_infer.hpp"
#include "common/model_info.hpp"
#include "common/utils.hpp"
#include "common/cv_cpp_utils.hpp"
#include "common/memory.hpp"
#include "pre_process/pre_process.cuh"
#include "post_process/post_process.cuh"

namespace tensorrt_infer
{
    namespace rtdetr_cuda
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class RTDETRDetect
        {
        public:
            RTDETRDetect() = default;
            ~RTDETRDetect();
            void initParameters(const std::string &engine_file, float score_thr = 0.5f); // 初始化参数
            void adjust_memory(int batch_size);                                          // 由于batch size是动态的，所以需要对gpu/cpu内存进行动态的申请

            // forward
            BoxArray forward(const Image &image);
            BatchBoxArray forwards(const std::vector<Image> &images);

            // 模型前后处理
            void preprocess_gpu(int ibatch, const Image &image,
                                shared_ptr<Memory<unsigned char>> preprocess_buffer, cudaStream_t stream_);
            void postprocess_gpu(int ibatch, cudaStream_t stream_);
            BatchBoxArray parser_box(const std::vector<Image> &images);

            // draw image
            void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir);
            void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir);

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            // 使用自定义的Memory类用来申请gpu/cpu内存
            std::vector<std::shared_ptr<Memory<unsigned char>>> preprocess_buffers_;
            Memory<float> input_buffer_, bbox_predict_, output_boxarray_;

            // 使用cuda流进行操作
            cudaStream_t cu_stream;

            // time
            Timer timer;
        };
    }
}

#endif // _RTDETR_DETECT_CUDA_HPP_