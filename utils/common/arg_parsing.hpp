#ifndef UTILS_ARG_PARSING_H_
#define UTILS_ARG_PARSING_H_
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <getopt.h>

#define RETURN_SUCCESS (0)
#define RETURN_FAIL (-1)
namespace ai
{
    namespace arg_parsing
    {
        struct Settings
        {
            std::string model_path = "";     // 用来接收命令行传递的模型路径
            std::string image_path = "";     // 用来接收命令行传递的要推理图片的路径
            std::string device_type = "gpu"; // 用于gpu、cpu推理，注：如果是tensorrt推理引擎cpu是指前后处理是cpu

            int batch_size = 1;     // 模型推理时需要batch_size张图片同时推理
            float score_thr = 0.5f; // 模型结果筛选常用到的阈值
            int device_id = 0;      // 可通过命令行修改要运行的显卡id，注:如果你是多gpu
            int loop_count = 10;    // 推理任务循环跑的次数，多计算推理时间
            // int number_of_threads = 4;
            int number_of_warmup_runs = 2; // 模型推理的预热，用来激活cuda核，是的计时更加准确
            std::string output_dir = "";   // 模型推理结果图片存储位置
        };
        int parseArgs(int argc, char **argv, Settings *s); // 解析命令行输入的参数并赋值给Settings
        void printArgs(Settings *s);                       // 打印所有的参数
    }
}

#endif // UTILS_ARG_PARSING_H_