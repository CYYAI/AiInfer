#include <string>
#include <opencv2/opencv.hpp>
#include "common/arg_parsing.hpp"
#include "common/cv_cpp_utils.hpp"
#include "yolo_series_app/yolo_det_cuda/yolo_series_detect.hpp"

void trt_cuda_inference(ai::arg_parsing::Settings *s, const std::string &model_name)
{
    // 首先判断一下输入的modle_name是否正确
    if (strcmp(model_name.c_str(), "yolov5") && strcmp(model_name.c_str(), "yolox") && strcmp(model_name.c_str(), "yolov6") && strcmp(model_name.c_str(), "yolov7"))
    {
        INFO("yolo series not support %s model infer\n", model_name.c_str());
        exit(RETURN_FAIL);
    }

    ai::utils::Timer timer; // 创建
    tensorrt_infer::yolo_series_cuda::YOLOSeriesDetect yolo_series_obj;
    yolo_series_obj.initParameters(model_name, s->model_path, s->score_thr);

    // 判断图片路径是否存在
    if (!ai::utils::file_exist(s->image_path))
    {
        INFO("Error: image path is not exist!!!");
        exit(RETURN_FAIL);
    }

    // 加载要推理的数据
    std::vector<cv::Mat> images;
    for (int i = 0; i < s->batch_size; i++)
        images.push_back(cv::imread(s->image_path));
    std::vector<ai::cvUtil::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), ai::cvUtil::cvimg_trans_func);

    // 模型预热，如果要单张推理，请调用yolov8_obj.forward
    for (int i = 0; i < s->number_of_warmup_runs; ++i)
        auto warmup_batched_result = yolo_series_obj.forwards(yoloimages);

    ai::cvUtil::BatchBoxArray batched_result;
    // 模型推理
    timer.start();
    for (int i = 0; i < s->loop_count; ++i)
        batched_result = yolo_series_obj.forwards(yoloimages);
    timer.stop(ai::utils::path_join("Batch=%d, iters=%d,run infer mean time:", s->batch_size, s->loop_count).c_str(), s->loop_count);

    if (!s->output_dir.empty())
    {
        ai::utils::rmtree(s->output_dir);
        ai::cvUtil::draw_batch_rectangle(images, batched_result, s->output_dir, s->classlabels);
    }
}

int main(int argc, char *argv[])
{
    ai::arg_parsing::Settings s;
    if (parseArgs(argc, argv, &s) == RETURN_FAIL)
    {
        INFO("Failed to parse the args\n");
        return RETURN_FAIL;
    }
    ai::arg_parsing::printArgs(&s);

    ///////////////////////////
    s.model_path = "weights/yolov7_dynamic_fp16.engine";
    s.image_path = "res/bus.jpg";
    s.output_dir = "cuda_res";
    ///////////////////////////

    CHECK(cudaSetDevice(s.device_id)); // 设置你用哪块gpu
    // 第二个参数只支持：yolov5,yolov6,yolox,yolov7
    // 由于只有这一个系列，所以该参数就不放到arg_parse里面了，根据你的需求选择或拆开即可
    trt_cuda_inference(&s, "yolov7"); // tensorrt的gpu版本推理：模型前处理和后处理都是使用cuda实现
    return RETURN_SUCCESS;
}