#include "cv_cpp_utils.hpp"
namespace ai
{
    namespace cvUtil
    {
        Image cvimg_trans_func(const cv::Mat &image)
        {
            return Image(image.data, image.cols, image.rows, image.channels());
        }

        Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type)
        {

            Norm out;
            out.type = NormType::MeanStd;
            out.alpha = alpha;
            out.channel_type = channel_type;
            memcpy(out.mean, mean, sizeof(out.mean));
            memcpy(out.std, std, sizeof(out.std));
            return out;
        }

        Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type)
        {

            Norm out;
            out.type = NormType::AlphaBeta;
            out.alpha = alpha;
            out.beta = beta;
            out.channel_type = channel_type;
            return out;
        }

        Norm Norm::None()
        {
            return Norm();
        };

        void AffineMatrix::compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to)
        {
            float scale_x = get<0>(to) / (float)get<0>(from);
            float scale_y = get<1>(to) / (float)get<1>(from);
            float scale = std::min(scale_x, scale_y);
            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

            double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            D = D != 0. ? double(1.) / D : double(0.);
            double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
            double b1 = -A11 * i2d[2] - A12 * i2d[5];
            double b2 = -A21 * i2d[2] - A22 * i2d[5];

            d2i[0] = A11;
            d2i[1] = A12;
            d2i[2] = b1;
            d2i[3] = A21;
            d2i[4] = A22;
            d2i[5] = b2;
        }

        void draw_batch_rectangle(std::vector<cv::Mat> &images, BatchBoxArray &batched_result, const std::string &save_dir, const std::vector<std::string> &classlabels)
        {
            for (int ib = 0; ib < (int)batched_result.size(); ++ib)
            {
                auto &objs = batched_result[ib];
                auto &image = images[ib];
                for (auto &obj : objs)
                {
                    uint8_t b, g, r;
                    tie(b, g, r) = random_color(obj.class_label);
                    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                                  cv::Scalar(b, g, r), 2);
                    auto name = classlabels[obj.class_label];
                    auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                                16);
                }
                if (mkdirs(save_dir))
                {
                    std::string save_path = path_join("%s/Infer_%d.jpg", save_dir.c_str(), ib);
                    cv::imwrite(save_path, image);
                }
            }
        }

        void draw_one_image_rectangle(cv::Mat &image, BoxArray &result, const std::string &save_dir, const std::vector<std::string> &classlabels)
        {

            for (auto &obj : result)
            {
                uint8_t b, g, r;
                tie(b, g, r) = random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                              cv::Scalar(b, g, r), 2);
                auto name = classlabels[obj.class_label];
                auto caption = cv::format("%s %.2f", name.c_str(), obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                              cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
                            16);
            }
            if (mkdirs(save_dir))
            {
                std::string save_path = path_join("%s/Infer_one.jpg", save_dir.c_str());
                cv::imwrite(save_path, image);
            }
        }
    }
}
