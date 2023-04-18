rm -rf build && mkdir -p build && cd build && cmake .. && make -j48 && cd ..
cd workspaces
./infer -f weights/yolov8_face_mask_detect_0.951_int8_quantized_dynamic.engine -i res/demo_h.jpg -b 10 -c 10 -w 5 -d gpu -g 2 -o cuda_res
./infer -f weights/yolov8_face_mask_detect_0.951_int8_quantized_dynamic.engine -i res/demo_h.jpg -b 10 -c 10 -w 5 -d cpu -g 2 -o cpp_res
cd ..