mkdir -p build && cd build && cmake .. && make -j8 && cd ..
cd workspaces
./infer -f weights/yolov8_face_mask_detect_0.951_int8_quantized_dynamic.engine -i res/test_demo.jpg -b 10 -c 5000000 -w 5 -g 2 -o result_output
cd ..