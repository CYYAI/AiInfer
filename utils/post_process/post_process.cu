#include "post_process.cuh"
namespace ai
{
    namespace postprocess
    {
        // keepflag主要是用来进行nms时候判断是否将该框抛弃
        const int NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag
        static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
        {
            *ox = matrix[0] * x + matrix[1] * y + matrix[2];
            *oy = matrix[3] * x + matrix[4] * y + matrix[5];
        }

        static __global__ void decode_kernel_common(float *predict, int num_bboxes, int num_classes,
                                                    int output_cdim, float confidence_threshold,
                                                    float *invert_affine_matrix, float *parray,
                                                    int MAX_IMAGE_BOXES)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            // pitem就获取了每个box的首地址
            // output_cdim是指每个box中有几个元素，可以根据onnx最后的输出定
            // 每个box的元素一般是(根据你的模型可修改,objectness是yolo系列的是否有物体得分)：left,top,right,bottom,objectness,class0,class1,...,classn
            float *pitem = predict + output_cdim * position;
            float objectness = pitem[4];
            if (objectness < confidence_threshold)
                return;

            // 从多个类别得分中，找出最大类别的class_score+label
            float *class_confidence = pitem + 5;
            float confidence = *class_confidence++; // 取class1给confidence并且class_confidence自增1
            int label = 0;
            // ++class_confidence和class_confidence++在循环中执行的结果是一样的，都是执行完循环主体后再加一
            for (int i = 1; i < num_classes; ++i, ++class_confidence)
            {
                if (*class_confidence > confidence)
                {
                    confidence = *class_confidence;
                    label = i;
                }
            }

            confidence *= objectness; // yolo系列的最终得分是两者相乘
            if (confidence < confidence_threshold)
                return;

            // cuda的原子操作：int atomicAdd(int *M,int V); 它们把一个内存位置M和一个数值V作为输入。
            // 与原子函数相关的操作在V上执行，数值V早已存储在内存地址*M中了，然后将相加的结果写到同样的内存位置中。
            int index = atomicAdd(parray, 1); // 所以这段代码意思是用parray[0]来计算boxes的总个数
            if (index >= MAX_IMAGE_BOXES)
                return;

            float cx = *pitem++;
            float cy = *pitem++;
            float width = *pitem++;
            float height = *pitem++;
            float left = cx - width * 0.5f;
            float top = cy - height * 0.5f;
            float right = cx + width * 0.5f;
            float bottom = cy + height * 0.5f;
            // boxes映射回相对于真实图片的尺寸
            affine_project(invert_affine_matrix, left, top, &left, &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

            // parray+1之后的值全部用来存储boxes元素，每个框有NUM_BOX_ELEMENT个元素
            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
        }

        static __device__ float box_iou(
            float aleft, float atop, float aright, float abottom,
            float bleft, float btop, float bright, float bbottom)
        {

            float cleft = max(aleft, bleft);
            float ctop = max(atop, btop);
            float cright = min(aright, bright);
            float cbottom = min(abottom, bbottom);

            float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
            if (c_area == 0.0f)
                return 0.0f;

            float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
            float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
            return c_area / (a_area + b_area - c_area);
        }

        static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold)
        {

            int position = (blockDim.x * blockIdx.x + threadIdx.x);
            int count = min((int)*bboxes, max_objects);
            if (position >= count)
                return;

            // left, top, right, bottom, confidence, class, keepflag
            float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
            for (int i = 0; i < count; ++i)
            {
                float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
                if (i == position || pcurrent[5] != pitem[5])
                    continue;

                if (pitem[4] >= pcurrent[4])
                {
                    if (pitem[4] == pcurrent[4] && i < position)
                        continue;

                    float iou = box_iou(
                        pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                        pitem[0], pitem[1], pitem[2], pitem[3]);

                    if (iou > threshold)
                    {
                        pcurrent[6] = 0; // 1=keep, 0=ignore
                        return;
                    }
                }
            }
        }

        static __global__ void decode_kernel_v8_trans(float *predict, int num_bboxes, int num_classes,
                                                      int output_cdim, float confidence_threshold,
                                                      float *invert_affine_matrix, float *parray,
                                                      int MAX_IMAGE_BOXES)
        {
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= num_bboxes)
                return;

            // yolov8和其他yolo系列的box不一样，是：left,top,right,bottom,class0,class1,...,classn
            // 然后在class0,class1,...,classn中取最大的座位score和label，去除了objectness，且是列排序，所以，需要对前面的解析代码稍微改变
            // float *pitem = predict + output_cdim * position;
            float max_confidence = *(predict + 4 * num_bboxes + position);
            int label = 0;
            for (int i = 1; i < num_classes; ++i)
            {
                if (*(predict + (4 + i) * num_bboxes + position) > max_confidence)
                {
                    max_confidence = *(predict + (4 + i) * num_bboxes + position);
                    label = i;
                }
            }
            if (max_confidence < confidence_threshold)
                return;

            int index = atomicAdd(parray, 1);
            if (index >= MAX_IMAGE_BOXES)
                return;

            float cx = *(predict + 0 * num_bboxes + position);
            float cy = *(predict + 1 * num_bboxes + position);
            float width = *(predict + 2 * num_bboxes + position);
            float height = *(predict + 3 * num_bboxes + position);
            float left = cx - width * 0.5f;
            float top = cy - height * 0.5f;
            float right = cx + width * 0.5f;
            float bottom = cy + height * 0.5f;
            affine_project(invert_affine_matrix, left, top, &left, &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

            float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = max_confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
        }

        void decode_detect_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                          float confidence_threshold, float *invert_affine_matrix,
                                          float *parray, int MAX_IMAGE_BOXES, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);

            checkCudaKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
                predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
                parray, MAX_IMAGE_BOXES));
        }

        void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, cudaStream_t stream)
        {

            auto grid = CUDATools::grid_dims(max_objects);
            auto block = CUDATools::block_dims(max_objects);
            checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
        }

        void decode_detect_yolov8_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, float *invert_affine_matrix,
                                                 float *parray, int MAX_IMAGE_BOXES, cudaStream_t stream)
        {
            auto grid = CUDATools::grid_dims(num_bboxes);
            auto block = CUDATools::block_dims(num_bboxes);
            // yolov3/v5/v7/yolox等模型的输出格式是[batch,num_boxes,output_cdim],这样，每个框的所有值是连续排列的[行排序]，方便使用
            // 但是yolov8的输出是[batch,output_cdim,num_boxes],这就超难受了，每个框的所以值都不连续，冲突是最大的[列排序]，解决方案:
            // 1. 从onnx的导出上解决，直接将其维度[batch,6,8400]-->[batch,8400,6],这样再生成engine就可以了，这个速度较快
            // 2. 从解析结果层面解决，但这会造成kernel函数执行线程存储体的冲突且冲突是最大的，所以这个方法速度稍慢，本节用这个
            checkCudaKernel(decode_kernel_v8_trans<<<grid, block, 0, stream>>>(
                predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
                parray, MAX_IMAGE_BOXES));
        }
    }
}