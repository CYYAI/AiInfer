import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import os

# custom functional package
import bevfusion_qat.bevcore.exptool as exptool
import bevfusion_qat.bevcore.quantize as quantize


def parse_arg():
    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel",
                        type=int,
                        default=5,
                        help="SCN num of input channels")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="bevfusion_qat/quant_model/ckpt/bevfusion_ptq.pth",
        help="SCN Checkpoint (scn backbone checkpoint)")
    parser.add_argument("--save_onnx_dir",
                        type=str,
                        default="bevfusion_qat/quant_model",
                        help="output onnx")
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Transfer the coordinate order of the index from xyz to zyx")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    inverse_indices = args.inverse
    if inverse_indices:
        save_onnx_path = os.path.join(args.save_onnx_dir,
                                      "lidar.backbone.zyx.onnx")
    else:
        save_onnx_path = os.path.join(args.save_onnx_dir,
                                      "lidar.backbone.xyz.onnx")

    os.makedirs(os.path.dirname(save_onnx_path), exist_ok=True)
    model = torch.load(args.ckpt).module
    model.eval().cuda().half()
    model = model.encoders.lidar.backbone

    quantize.disable_quantization(model).apply()

    # Set layer attributes
    for name, module in model.named_modules():
        module.precision = "int8"
        module.output_precision = "int8"

    model.conv_input.precision = "fp16"
    model.conv_out.output_precision = "fp16"

    voxels = torch.zeros(1, args.in_channel).cuda().half()
    coors = torch.zeros(1, 4).int().cuda()
    batch_size = 1

    exptool.export_onnx(model, voxels, coors, batch_size, inverse_indices,
                        save_onnx_path)
