#include "yolov8_detect.hpp"
namespace tensorrt_infer
{
    namespace yolov8_cpp
    {
        void YOLOv8Detect::initParameters(const std::string &engine_file, float score_thr, float nms_thr)
        {
            if (!file_exist(engine_file))
            {
                INFO("Error: engine_file is not exist!!!");
                exit(0);
            }

            this->model_info = std::make_shared<ModelInfo>();
            // 传入参数的配置
            model_info->m_modelPath = engine_file;
            model_info->m_postProcCfg.confidence_threshold_ = score_thr;
            model_info->m_postProcCfg.nms_threshold_ = nms_thr;
            this->model_ = trt::infer::load(engine_file); // 加载infer对象
            this->model_->print();                        // 打印engine的一些基本信息

            // 获取输入的尺寸信息
            auto input_dim = this->model_->get_network_dims(0); // 获取输入维度信息
            model_info->m_preProcCfg.infer_batch_size = input_dim[0];
            model_info->m_preProcCfg.network_input_channels_ = input_dim[1];
            model_info->m_preProcCfg.network_input_height_ = input_dim[2];
            model_info->m_preProcCfg.network_input_width_ = input_dim[3];
            model_info->m_preProcCfg.network_input_numel = input_dim[1] * input_dim[2] * input_dim[3];
            model_info->m_preProcCfg.isdynamic_model_ = this->model_->has_dynamic_dim();
            // 对输入的图片预处理进行配置,即，yolov8的预处理是除以255，并且是RGB通道输入
            model_info->m_preProcCfg.normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::RGB);

            // 获取输出的尺寸信息
            auto output_dim = this->model_->get_network_dims(1);
            model_info->m_postProcCfg.bbox_head_dims_ = output_dim;
            model_info->m_postProcCfg.bbox_head_dims_output_numel_ = output_dim[1] * output_dim[2];
            if (model_info->m_postProcCfg.num_classes_ == 0)
                model_info->m_postProcCfg.num_classes_ = model_info->m_postProcCfg.bbox_head_dims_[1] - 4; // yolov8
            model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT = model_info->m_postProcCfg.MAX_IMAGE_BOXES * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
            CHECK(cudaStreamCreate(&cu_stream)); // 创建cuda流
        }

        YOLOv8Detect::~YOLOv8Detect()
        {
            CHECK(cudaStreamDestroy(cu_stream)); // 销毁cuda流
        }

        void YOLOv8Detect::adjust_memory(int batch_size)
        {
            // 申请模型输入和模型输出所用到的内存
            input_buffer_.cpu(batch_size * model_info->m_preProcCfg.network_input_numel);           // 申请batch个模型输入的cpu内存
            input_buffer_.gpu(batch_size * model_info->m_preProcCfg.network_input_numel);           // 申请batch个模型输入的gpu内存
            bbox_predict_.cpu(batch_size * model_info->m_postProcCfg.bbox_head_dims_output_numel_); // 申请batch个模型输出的cpu内存
            bbox_predict_.gpu(batch_size * model_info->m_postProcCfg.bbox_head_dims_output_numel_); // 申请batch个模型输出的gpu内存

            // 申请模型解析成box时需要存储的内存,,+32是因为第一个数要设置为框的个数，防止内存溢出
            output_boxarray_.cpu(batch_size * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT));
        }

        std::tuple<float, int, int> YOLOv8Detect::preprocess_cpu(int ibatch, const Image &image)
        {
            if (image.channels != model_info->m_preProcCfg.network_input_channels_)
            {
                INFO("Warning : Number of channels wanted differs from number of channels in the actual image \n");
                exit(-1);
            }

            // 模型输入的cpu指针，通过yolov8_preprocess_image函数获取值
            float *model_input_host = input_buffer_.cpu() + ibatch * model_info->m_preProcCfg.network_input_numel;

            // 调用yolov8 cpu版本的的前处理
            return yolov8_preprocess_image(image, model_input_host, model_info->m_preProcCfg, const_value);
        }

        void YOLOv8Detect::postprocess_cpu(int ibatch)
        {
            float scale = std::get<0>(pad_scale_vec[ibatch]);
            int fill_width = std::get<1>(pad_scale_vec[ibatch]);
            int fill_height = std::get<2>(pad_scale_vec[ibatch]);
            // boxarray_host：对推理结果进行解析后要存储的cpu指针
            float *boxarray_host = output_boxarray_.cpu() + ibatch * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
            memset(boxarray_host, 0, sizeof(int)); // 将首元素设置为0
            // image_based_bbox_output:推理结果产生的所有预测框的cpu指针
            float *image_based_bbox_output = bbox_predict_.cpu() + ibatch * model_info->m_postProcCfg.bbox_head_dims_output_numel_;

            yolov8_decode_network_out_trans(image_based_bbox_output, model_info->m_postProcCfg.bbox_head_dims_[2], model_info->m_postProcCfg.num_classes_,
                                            model_info->m_postProcCfg.bbox_head_dims_[1], model_info->m_postProcCfg.confidence_threshold_,
                                            boxarray_host, model_info->m_postProcCfg.MAX_IMAGE_BOXES, model_info->m_postProcCfg.NUM_BOX_ELEMENT,
                                            scale, fill_width, fill_height);

            // 对筛选后的框进行nms操作
            cpu_nms(boxarray_host, model_info->m_postProcCfg);
        }

        BatchBoxArray YOLOv8Detect::parser_box(int num_image)
        {
            BatchBoxArray arrout(num_image);
            for (int ib = 0; ib < num_image; ++ib)
            {
                float *parray = output_boxarray_.cpu() + ib * (32 + model_info->m_postProcCfg.IMAGE_MAX_BOXES_ADD_ELEMENT);
                int count = min(model_info->m_postProcCfg.MAX_IMAGE_BOXES, (int)*parray);
                BoxArray &output = arrout[ib];
                output.reserve(count); // 增加vector的容量大于或等于count的值
                for (int i = 0; i < count; ++i)
                {
                    float *pbox = parray + 1 + i * model_info->m_postProcCfg.NUM_BOX_ELEMENT;
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1)
                        // left,top,right,bottom,confidence,class_label
                        output.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                }
            }

            return arrout;
        }

        BoxArray YOLOv8Detect::forward(const Image &image)
        {
            auto output = forwards({image});
            if (output.empty())
                return {};
            return output[0];
        }

        BatchBoxArray YOLOv8Detect::forwards(const std::vector<Image> &images)
        {
            int num_image = images.size();
            if (num_image == 0)
                return {};
            pad_scale_vec.clear();

            // 动态设置batch size
            auto input_dims = model_->get_network_dims(0);
            if (model_info->m_preProcCfg.infer_batch_size != num_image)
            {
                if (model_info->m_preProcCfg.isdynamic_model_)
                {
                    model_info->m_preProcCfg.infer_batch_size = num_image;
                    input_dims[0] = num_image;
                    if (!model_->set_network_dims(0, input_dims)) // 重新绑定输入batch，返回值类型是bool
                        return {};
                }
                else
                {
                    if (model_info->m_preProcCfg.infer_batch_size < num_image)
                    {
                        INFO(
                            "When using static shape model, number of images[%d] must be "
                            "less than or equal to the maximum batch[%d].",
                            num_image, model_info->m_preProcCfg.infer_batch_size);
                        return {};
                    }
                }
            }

            // 由于batch size是动态的，所以需要对gpu/cpu内存进行动态的申请
            adjust_memory(model_info->m_preProcCfg.infer_batch_size);

            // 对图片进行预处理
            for (int i = 0; i < num_image; ++i)
                pad_scale_vec.push_back(preprocess_cpu(i, images[i])); // input_buffer_会获取到图片预处理好的值

            // 由于tensorrt推理只能在gpu上推理，所以模型输入只能传递到gpu内存上
            checkRuntime(cudaMemcpyAsync(input_buffer_.gpu(), input_buffer_.cpu(),
                                         input_buffer_.cpu_bytes(), cudaMemcpyHostToDevice, cu_stream));

            // 推理模型
            float *bbox_output_device = bbox_predict_.gpu();                  // 获取推理后要存储结果的gpu指针
            vector<void *> bindings{input_buffer_.gpu(), bbox_output_device}; // 绑定bindings作为输入进行forward
            if (!model_->forward(bindings, cu_stream))
            {
                INFO("Failed to tensorRT forward.");
                return {};
            }

            // 模型的推理结果是在gpu上，由于后处理要经过cpp版的nms，所以要将结果从gpu传递到gpu
            checkRuntime(cudaMemcpyAsync(bbox_predict_.cpu(), bbox_predict_.gpu(),
                                         bbox_predict_.gpu_bytes(), cudaMemcpyDeviceToHost, cu_stream));
            checkRuntime(cudaStreamSynchronize(cu_stream)); // 阻塞异步流，等流中所有操作执行完成才会继续执行

            // 对推理结果进行解析
            for (int ib = 0; ib < num_image; ++ib)
                postprocess_cpu(ib);

            return parser_box(num_image);
        }
    }
}