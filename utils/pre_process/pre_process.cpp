#include "pre_process.hpp"
namespace ai
{
    namespace preprocess
    {
        std::tuple<float, int, int> yolov8_preprocess_image(const Image &image, float *dst, PreprocessImageConfig preProcessImageConfig, const uint8_t const_value)
        {
            float r = std::min(preProcessImageConfig.network_input_width_ / (image.width * 1.0),
                               preProcessImageConfig.network_input_height_ / (image.height * 1.0));
            int unpad_w = r * image.width;
            int unpad_h = r * image.height;
            cv::Mat re(unpad_h, unpad_w, CV_8UC3);
            cv::Mat src_image(image.height, image.width, CV_8UC3, const_cast<void *>(image.bgrptr));
            cv::resize(src_image, re, re.size());
            cv::Mat out(preProcessImageConfig.network_input_height_,
                        preProcessImageConfig.network_input_width_,
                        CV_8UC3, cv::Scalar(const_value, const_value, const_value));

            int fill_height = (int)(preProcessImageConfig.network_input_height_ - unpad_h) / 2;
            int fill_width = (int)(preProcessImageConfig.network_input_width_ - unpad_w) / 2;
            re.copyTo(out(cv::Rect(fill_width, fill_height, re.cols, re.rows)));

            uint8_t *pSrc;
            cv::Mat spl[preProcessImageConfig.network_input_channels_];
            cv::split(out, spl);
            Norm norm = preProcessImageConfig.normalize_;
            int network_wh_numel = preProcessImageConfig.network_input_width_ * preProcessImageConfig.network_input_height_;

            for (int j = 0; j < preProcessImageConfig.network_input_channels_; j++)
            {

                if (norm.channel_type == ChannelType::RGB)
                    pSrc = (uint8_t *)spl[preProcessImageConfig.network_input_channels_ - (j + 1)].data;
                else if (norm.channel_type == ChannelType::BGR)
                    pSrc = (uint8_t *)spl[j].data;
                for (int i = 0; i < network_wh_numel; i++)
                {
                    if (norm.type == NormType::MeanStd)
                        dst[j * network_wh_numel + i] = ((float)pSrc[i] * norm.alpha - norm.mean[j]) / norm.std[j];
                    else if (norm.type == NormType::AlphaBeta)
                        dst[j * network_wh_numel + i] = (float)pSrc[i] * norm.alpha + norm.beta;
                }
            }

            return std::make_tuple(r, fill_width, fill_height);
        }
    }
}