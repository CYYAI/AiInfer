#ifndef _YOLOV8_FACE_DETECT_CPP_HPP_
#define _YOLOV8_FACE_DETECT_CPP_HPP_
#include <memory>
#include "backend/tensorrt/trt_infer.hpp"
#include "common/model_info.hpp"
#include "common/utils.hpp"
#include "common/cv_cpp_utils.hpp"
#include "common/memory.hpp"
#include "pre_process/pre_process.hpp"
#include "post_process/post_process.hpp"

namespace tensorrt_infer
{
    namespace yolov8_cpp
    {
        using namespace ai::modelInfo;
        using namespace ai::utils;
        using namespace ai::cvUtil;
        using namespace ai::memory;
        using namespace ai::preprocess;
        using namespace ai::postprocess;

        class YOLOv8Detect
        {
        public:
            YOLOv8Detect() = default;
            ~YOLOv8Detect();
            void initParameters(const std::string &engine_file, float score_thr = 0.5f,
                                float nms_thr = 0.45f); // 初始化参数
            void adjust_memory(int batch_size);         // 由于batch size是动态的，所以需要对gpu/cpu内存进行动态的申请

            // forward
            BoxArray forward(const Image &image);
            BatchBoxArray forwards(const std::vector<Image> &images);

            // 模型前后处理
            std::tuple<float, int, int> preprocess_cpu(int ibatch, const Image &image);
            void postprocess_cpu(int ibatch);
            BatchBoxArray parser_box(int num_image);

            // draw image
            void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir);
            void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir);

        private:
            std::shared_ptr<ai::backend::Infer> model_;
            std::shared_ptr<ModelInfo> model_info = nullptr;

            // 仿射矩阵的声明
            const uint8_t const_value = 114; // 图片resize补边时的值

            // 收集每张图片的缩放尺寸
            std::vector<std::tuple<float, int, int>> pad_scale_vec;

            // 使用自定义的Memory类用来申请gpu/cpu内存
            Memory<float> input_buffer_, bbox_predict_, output_boxarray_;

            // 使用cuda流进行操作
            cudaStream_t cu_stream;
        };
    }
}

#endif // _YOLOV8_FACE_DETECT_CPP_HPP_