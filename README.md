## 最新消息
该项目已废弃，推荐本人参与的推理框架项目：[nndeploy](https://github.com/DeployAI/nndeploy)，欢迎各位成为开发者

## 项目介绍
这是一个c++版的AI推理库，目前只支持tensorrt模型的推理，后续计划支持Openvino、NCNN、MNN等框架的c++推理。前后处理提供两个版本，c++版和cuda版，建议使用cuda版。

## 新增项目消息:
- 🔥增加了RT-DETR的目标检测tensorrt推理
    - [导出RT-DETR-Engine模型教程](https://zhuanlan.zhihu.com/p/623794029)
    - [RT-DETR检测cuda版本](application/rtdetr_det_app/rtdetr_cuda)

- 🔥增加了多目标追踪**ByteTrack**的实现(速度快,精度高)，强烈推荐
    - [ByteTrack的使用教程](application/det_track_app/README.md)

- 🔥增加了yolov8各个任务的tensorrt推理，包含检测、分割、姿态估计
    - [导出YOLOv8-Engine模型教程](application/yolov8_app/README.md)
    - [yolov8-detection cuda版本](application/yolov8_app/yolov8_det_cuda)
    - [yolov8-segment cuda版本](application/yolov8_app/yolov8_seg_cuda)
    - [yolov8-pose cuda版本](application/yolov8_app/yolov8_pose_cuda)

- 🔥增加yolo系列通用的检测代码，包含yolov5、yolox、yolov6、yolov7
    - [导出各yolo系列Engine模型的教程](application/yolo_series_app/README.md)
    - [上述yolo系列通用det-cuda代码](application/yolo_series_app)
    
## 其他backend推理代码
- [ Openvino ] coming soon
- [ NCNN ] coming soon
- [ MNN ] coming soon
- 本来打算直接写成一个项目，但这样项目就会较大，较冗余，经群友建议，后面会在这个项目中建立其他分支来实现。

## 项目目录介绍
```bash
AiInfer
  |--application # 模型推理应用的实现，你自己的模型推理可以在该目录下实现
    |--yolov8_det_app # 举例：实现的一个yolov8检测
    |--xxxx
  |--utils # 工具目录
    |--backend # 这里实现backend的推理类
    |--common # 里面放着一些常用的工具类
      |--arg_parsing.hpp # 命令行解析类，类似python的argparse
      |--cuda_utils.hpp # 里面放着一些cuda常用的工具函数
      |--cv_cpp_utils.hpp # 里面放着一些cv相关的工具函数
      |--memory.hpp # 有关cpu、gpu内存申请和释放的工具类
      |--model_info.hpp # 有关模型的前后处理的常用参数定义，例如均值方差、nms阈值等
      |--utils.hpp # cpp中常用到的工具函数，计时、mkdir等
    |--post_process # 后处理实现目录，cuda后处理加速,如果你有自定义的后处理也可以写在这里
    |--pre_process # 前处理实现目录，cuda前处理加速,如果你有自定义的前处理也可以写在这里
    |--tracker # 这个是目标检测追踪库的实现，已解耦，不想用可直接删除
  |--workspaces # 工作目录，里面可以放一些测试图片/视频、模型，然后在main.cpp中直接使用相对路径
  |--mains # 这里面是main.cpp合集，这里采用每个app单独对应一个main文件，便于理解，写一起太冗余
```

## 如何开始
<details>
<summary>1. Linux & Windows下环境配置</summary>

- linux推荐使用VSCode,windows推荐使用visual studio 2019
- 安装显卡驱动、cuda、cudnn、opencv、tensorrt-->[安装教程](https://zhuanlan.zhihu.com/p/624170244)
- 建议先从一个检测的例子入手，熟悉项目架构，例如：application/yolov8_app/yolov8_det_cuda

</details>

<details>
<summary>2. onnx转trt【fp16+int8】</summary>

- onnx的导出建议是动态batch，这里举例pytorch模型的导出，如果你需要动态宽高，该项目也支持~
```python
torch.onnx._export(
        model,
        dummy_input, # 例如torch.randn(1,3,640,640)
        save_onnx_path,
        input_names=["image"],
        output_names=["output"],
        dynamic_axes={'image': {0: 'batch'},
                      'output': {0: 'batch'}},
        opset_version=args.opset, # 一般11或12更加适用于各种芯片或板子
    )
```
- 将onnx精简[可选]
```bash
# 注意，如果你已经在代码中运行过onnxsim了，那就略过这步
pip install onnxsim # 安装onnxsim库，可以直接将复杂的onnx转为简单的onnx模型，且不改变其推理精度
onnxsim input_onnx_model output_onnx_model # 通过该命令行你会得到一个去除冗余算子的onnx模型
```
- onnx的fp16量化，转tensorrt，建议动态batch
```bash
# 前提，保证导出的onnx是动态batch，也就是输入shape是[-1,3,640,640]。注:640只是举例,输入你的宽高即可
trtexec --onnx=xxx_dynamic.onnx \
        --workspace=4098 \
        --minShapes=image:1x3x640x640 \
        --maxShapes=image:16x3x640x640 \
        --optShapes=image:4x3x640x640 \
        --saveEngine=xxx.engine \
        --avgRuns=100 \
        --fp16
```
- onnx的int8量化，这个尽量不要用trtexec导出，精度会有点问题，建议使用
  - [商汤的ppq的int8量化工具,支持tensorrt|openvino|mnn|ncnn|...](https://github.com/openppl-public/ppq)
  - [ppq不会使用的看yolov6的量化教程:](https://github.com/meituan/YOLOv6/tree/main/tools/quantization/ppq)
</details>

<details>
<summary>3. 项目编译和运行</summary>

- 配置CMakeLists中的计算能力为你的显卡对应值
    - 例如`-gencode=arch=compute_75,code=sm_75`，例如RTX3090是86，则是：`-gencode=arch=compute_86,code=sm_86`
    - 计算能力根据型号参考这里查看：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
    - 也可直接运行脚本[python3 assets/get_device_sm.py](assets/get_device_sm.py)获取sm值
- 在CMakeLists.txt中配置你本机安装的tensorrt路径，和add_executable中你要使用的main.cpp文件
- CMake:
    - `mkdir build && cd build`
    - `cmake ..`
    - `make -j8`
    - `cd ..`
-  查看项目需要输入的命令

```bash
cd workspaces
./infer -h
```
- --model_path, -f: 要输如模型的路径，必选
- --image_path, -i: 要输出的测试图片，必选
- --batch_size, -b: 要使用的batch_size[>=1]，可选，默认=1
- --score_thr, -s: 一般指后处理要筛选的得分阈值，可选，默认=0.5f
- --device_id, -g: 多显卡的显卡id,可选，默认=0
- --loop_count, -c: 要推理的次数，一般用于计时，可选，默认=10
- --warmup_runs, -w: 模型推理的预热次数(激活cuda核)，可选，默认=2
- --output_dir, -o: 要存储结果的目录，可选，默认=''
- --help, -h: 使用-h来查看都有哪些命令
```bash
# 然后运行按照你自己的要求运行即可，例如：
./infer -f xxx.engine -i xxx.jpg -b 10 -c 10 -o cuda_res # 使用cuda的前后处理，结果保存在cuda_res文件夹下
```
</details>

<details>
<summary>4. 制作成c++的sdk,交付项目</summary>

```bash
cd build
make install
# 然后你会在workspaces下看到一个install文件夹，这里面就是你要交付的include文件和so库
```
</details>

## B站同步视频讲解
- coming soon
## 附录
#### 1. qq联系我们，提提建议
![QQGroup](assets/infer_qq.png)
#### 2. 感谢相关项目
- https://github.com/meituan/YOLOv6
- https://github.com/openppl-public/ppq
- https://github.com/shouxieai/tensorRT_Pro
