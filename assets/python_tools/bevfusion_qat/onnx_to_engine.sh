# set parameters
precision=int8 # int8精度生成engine
base="bevfusion_qat/quant_model/onnx_int8" # 上步骤生成的onnx模型文件夹
engine_path="bevfusion_qat/quant_model/bevfusion_engines"

# precision flags
trtexec_fp16_flags="--fp16"
trtexec_dynamic_flags="--fp16"
if [ $precision==int8 ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

# onnx to engine function
    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
function compile_trt_model(){
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    onnx=$base/$name.onnx
    result_save_directory=$engine_path
    mkdir -p $result_save_directory

    if [ -f "${result_save_directory}/$name.engine" ]; then
        echo Model ${result_save_directory}/$name.engine already build 🙋🙋🙋.
        return
    fi

    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    # echo $number_of_input  $number_of_output

    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} ${output_flags} \
        --saveEngine=${result_save_directory}/$name.engine \
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed"
        # --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"

    echo Building the model: ${result_save_directory}/$name.engine, this will take several minutes. Wait a moment 🤗🤗🤗~.
    trtexec $cmd >> ${result_save_directory}/onnx_to_engine.log 2>&1
    if [ $? != 0 ]; then
        echo 😥 Failed to build model ${result_save_directory}/$name.engine.
        echo You can check the error message by ${result_save_directory}/$name.log 
        exit 1
    fi
}


# maybe int8 / fp16
compile_trt_model "camera.backbone" "$trtexec_dynamic_flags" 2 2
compile_trt_model "fuser" "$trtexec_dynamic_flags" 2 1

# fp16 only
compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1
compile_trt_model "head.bbox" "$trtexec_fp16_flags" 1 6