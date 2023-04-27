# rm -rf build && mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd workspaces
# ./infer -f weights/yolov8n_det_transd_dynamic_int8.engine -i res/dog.jpg -b 16 -c 300 -o cuda_res
# ./infer -f weights/yolov8s-seg_transd_dynamic_int8.engine -i res/dog.jpg -b 16 -c 300 -o cuda_res
# ./infer -f weights/yolov8s-pose_transd_dynamic_int8.engine -i res/bus.jpg -b 16 -c 3000 -o pose_res
# ./infer -f weights/rtdetr_r50vd_6x_coco_dynamic_int8.trt -i res/dog.jpg -b 16 -c 10 -o cuda_res
cd ..