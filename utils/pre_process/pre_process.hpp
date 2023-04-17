#ifndef __PRE_PROCESS_HPP_CPP_
#define __PRE_PROCESS_HPP_CPP_
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "common/cv_cpp_utils.hpp"
#include "common/model_info.hpp"

namespace ai
{
    namespace preprocess
    {
        using namespace ai::cvUtil;
        using namespace ai::modelInfo;

        std::tuple<float, int, int> yolov8_preprocess_image(const Image &image, float *dst, PreprocessImageConfig preProcessImageConfig, const uint8_t const_value);
    }
}
#endif // __PRE_PROCESS_HPP_CPP_