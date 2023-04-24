#ifndef _CV_CPP_UTILS_HPP_
#define _CV_CPP_UTILS_HPP_

#include <tuple>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

namespace ai
{
    namespace cvUtil
    {

        using namespace std;
        using namespace ai::utils;

        // 统一模型的输入格式，方便后续进行输入的配置
        struct Image
        {
            const void *bgrptr = nullptr;
            int width = 0, height = 0, channels = 0;

            Image() = default;
            Image(const void *bgrptr, int width, int height, int channels) : bgrptr(bgrptr), width(width), height(height), channels(channels) {}
        };

        Image cvimg_trans_func(const cv::Mat &image);

        // 对输入进行尺度缩放的flage配置
        enum class NormType : int
        {
            None = 0,
            MeanStd = 1,  // out = (x * alpha - mean) / std
            AlphaBeta = 2 // out = x * alpha + beta
        };

        // 设置输入通道是RGB还是BGR
        enum class ChannelType : int
        {
            BGR = 0,
            RGB = 1
        };

        // 可以通过该结构体来初始化对输入的配置
        struct Norm
        {
            float mean[3];
            float std[3];
            float alpha, beta;
            NormType type = NormType::None;
            ChannelType channel_type = ChannelType::BGR;

            // out = (x * alpha - mean) / std
            static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f, ChannelType channel_type = ChannelType::BGR);
            // out = x * alpha + beta
            static Norm alpha_beta(float alpha, float beta = 0.0f, ChannelType channel_type = ChannelType::BGR);
            // None
            static Norm None();
        };

        // 由于后面仿射变换使用cuda实现的，所以，这个结构体用来计算仿射变换的矩阵和逆矩阵
        struct AffineMatrix
        {
            float i2d[6]; // image to dst(network), 2x3 matrix
            float d2i[6]; // dst to image, 2x3 matrix
            void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to);
        };

        // detect
        struct Box
        {
            float left, top, right, bottom, confidence;
            int class_label;

            Box() = default;
            Box(float left, float top, float right, float bottom, float confidence, int class_label)
                : left(left),
                  top(top),
                  right(right),
                  bottom(bottom),
                  confidence(confidence),
                  class_label(class_label) {}
        };
        typedef std::vector<Box> BoxArray;
        typedef std::vector<BoxArray> BatchBoxArray;

        // draw image
        void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir, const std::vector<std::string> &classlabels);
        void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &classlabels);
    }
}

#endif // _CV_CPP_UTILS_HPP_