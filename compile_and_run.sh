# rm -rf build && mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd workspaces
./infer -f weights/rtdetr_r50vd_6x_coco_dynamic_fp16.trt -i res/dog.jpg -b 16 -c 10 -w 5 -d gpu -g 2 -o cuda_res
cd ..