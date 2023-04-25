mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd workspaces
./infer -f weights/rtdetr_r50vd_6x_coco_dynamic_fp16.trt -i res/dog.jpg -b 16 -c 100 -w 5 -d gpu -g 0 -o cuda_res_fp16
./infer -f weights/rtdetr_r50vd_6x_coco_dynamic_int8.trt -i res/dog.jpg -b 16 -c 100 -w 5 -d gpu -g 0 -o cuda_res_int8
cd ..