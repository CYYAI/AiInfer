rm -rf build && mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
# mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd workspaces
./infer -f weights/yolov8l_face_mask_detect_0.962_int8_trt_dynamic.engine -i res/demo_h.jpg -b 16 -c 10 -w 5 -d gpu -g 2 -o cuda_res
./infer -f weights/yolov8l_face_mask_detect_0.962_int8_trt_dynamic.engine -i res/demo_h.jpg -b 16 -c 10 -w 5 -d cpu -g 2 -o cpu_res
# ./infer -f weights/rtdetr_r50vd_6x_coco_dynamic_fp16.trt -i res/dog.jpg -b 16 -c 10 -w 5 -d gpu -g 2 -o cuda_res
cd ..