### 前言，该项目已支持的yolo代码库如下所示，请自行选择模型
- [yolov5官方代码仓库](https://github.com/ultralytics/yolov5)
- [yolox官方代码仓库](https://github.com/Megvii-BaseDetection/YOLOX)
- [yolov6官方代码仓库](https://github.com/meituan/YOLOv6)
- [yolov7官方代码仓库](https://github.com/WongKinYiu/yolov7)
- [mmyolo系列](https://github.com/open-mmlab/mmyolo)
- [paddleyolo系列](https://github.com/PaddlePaddle/PaddleYOLO)

### 上面模型的前处理后处理
```bash
# 上面四个模型的前处理大部分是相同的，但有的有些不同，具体区别如下
# 1. 前处理阶段
# 1.1 yolov5、yolov6、yolov7的前处理都是相同的，都是采用RGB通道，均值为0方差为1，并且各通道除以255
# 1.2 yolox的前处理是：采用BGR通道，均值为0，方差为1，各通道也不除255，这个需要注意一下
# 1.3 yolox的最前面层用了focus层，本项目实现了cuda版的resize+focus，你可以直接替换用以加速

# 2. 后处理阶段都是相同的，建议直接导出Detect层。一般而言，如果是coco数据集
  # 导出onnx的最后一层维度是：[batch,box_num,85]，box_num是所有预测框的总数，例如:yolox/yolov6是8400
  # 85分别指:cx,cy,width,height,obj_conf,cls_id0,cls_id1,...,cls_idn
```

## yolo-series onnx模型导出前的一些注意事项
- yolox模型：如果你使用的是yolox官方库，一定要将decode给导出，即
```bash
model.head.decode_in_inference = True # 必须设为True，要不然你需自己实现decode过程
```
- yolov7模型：yolov7官方库提供了nms一起导出到onnx，这里不建议，速度实测并不太快。建议使用cuda加速nms

### yolo-series中onnx的导出方法
```python
# 不仅仅是yolo系列，所有模型的导出都建议只动态batch导出，下面以pytorch导出动态batch的onnx举例：
torch.onnx._export(
        model,
        dummy_input, # 例如dummy_input=torch.randn(1,3,640,640)
        onnx_save_path,
        input_names=["images"], # 如果有多个输入就往列表中写
        output_names=["output"], # 如果有多个输出就往列表中写
        dynamic_axes={"images": {0: 'batch'}, # 我这里只写了动态batch，如果有多个输入或输出，直接添加即可
                      "output": {0: 'batch'}},
        opset_version=11, # 建议11/12，你如果部署过国内的一些板子，会发现他们的int8量化工具一般不支持较高版本的onnx，为了一个onnx走天下，建议这个版本不要太高
    )
# 视觉类项目大多宽高时静态的，当然，你要用到动态宽高，这个项目也支持~~~
```

### 将onnx精简[可选]
```bash
# 注意，如果你已经在代码中运行过onnxsim了，那就略过这步
pip install onnxsim # 安装onnxsim库，可以直接将复杂的onnx转为简单的onnx模型，且不改变其推理精度
onnxsim input_onnx_model output_onnx_model # 通过该命令行你会得到一个去除冗余算子的onnx模型
```

### yolo系列的onnx生成engine文件
- fp16量化生成的命令如下，这个精度损失不大，可以直接使用trtexec完成
```bash
trtexec --onnx=xxx_det.onnx \
        --minShapes=images:1x3x640x640 \
        --maxShapes=images:16x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --saveEngine=xxx_dynamic_fp16.engine \
        --avgRuns=100 \
        --fp16
```
- int8量化，这个直接用trtexec一般而言精度都有损失有的甚至无法工作，建议使用商汤ppq量化工具
  - [商汤的ppq的int8量化工具,支持tensorrt|openvino|mnn|ncnn|...](https://github.com/openppl-public/ppq)
  - [ppq不会使用的看yolov6的量化教程:](https://github.com/meituan/YOLOv6/tree/main/tools/quantization/ppq)

**生成Engine后直接编译运行即可，上面的代码都是经过测试的，如果出错请排查你的onnx或联系我解决**
- 程序入口：mains/main_yolo_series_det.cpp