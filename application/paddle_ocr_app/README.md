### 简介
OCR检测识别百度的[paddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)做的一直很不错，尤其是经过paddleslim的模型压缩，可以做到精度很高但模型很小，本次就演示PaddleOCR-v3的模型部署，包括：检测模型、文本识别模型、文本分类模型.

### 模型选择
本次使用PP-OCR V3的系列模型，[model zoo链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)，本次部署选择模型如下：
|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv3_det_slim|【最新】slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测|[ch_PP-OCRv3_det_cml.yml](../../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)| 1.1M |[文本检测推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar)|
|ch_PP-OCRv3_rec_slim |【最新】slim量化版超轻量模型，支持中英文、数字识别|[ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml)| 4.9M |[文本识别推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar)|
|ch_ppocr_mobile_slim_v2.0_cls|slim量化版模型，对检测到的文本行文字角度分类|[cls_mv3.yml](../../configs/cls/cls_mv3.yml)| 2.1M |[文本方向分类推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar) |
