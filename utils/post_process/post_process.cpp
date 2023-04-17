#include "post_process.hpp"
namespace ai
{
    namespace postprocess
    {
        void yolov8_decode_network_out_trans(float *predict, int num_bboxes, int num_classes, int output_cdim, float confidence_threshold,
                                             float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT,
                                             const float scale, const int fill_width, const int fill_height)
        {
            for (int position = 0; position < num_bboxes; position++)
            {
                // yolov8和其他yolo系列的box不一样，是：left,top,right,bottom,class0,class1,...,classn
                // 然后在class0,class1,...,classn中取最大的座位score和label，去除了objectness，且是列排序
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
                    continue;

                int index = *parray;
                parray[0] = index + 1;
                if (parray[0] >= MAX_IMAGE_BOXES)
                    return;

                float cx = *(predict + 0 * num_bboxes + position);
                float cy = *(predict + 1 * num_bboxes + position);
                float width = *(predict + 2 * num_bboxes + position);
                float height = *(predict + 3 * num_bboxes + position);
                float left = (cx - width * 0.5f - fill_width) / scale;
                float top = (cy - height * 0.5f - fill_height) / scale;
                float right = (cx + width * 0.5f - fill_width) / scale;
                float bottom = (cy + height * 0.5f - fill_height) / scale;

                float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
                *pout_item++ = left;
                *pout_item++ = top;
                *pout_item++ = right;
                *pout_item++ = bottom;
                *pout_item++ = max_confidence;
                *pout_item++ = label;
                *pout_item++ = 1; // 1 = keep, 0 = ignore
            }
        }

        float box_iou(float aleft, float atop, float aright, float abottom,
                      float bleft, float btop, float bright, float bbottom)
        {

            float cleft = std::max(aleft, bleft);
            float ctop = std::max(atop, btop);
            float cright = std::min(aright, bright);
            float cbottom = std::min(abottom, bbottom);

            float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
            if (c_area == 0.0f)
                return 0.0f;

            float a_area = std::max(0.0f, aright - aleft) * std::max(0.0f, abottom - atop);
            float b_area = std::max(0.0f, bright - bleft) * std::max(0.0f, bbottom - btop);
            return c_area / (a_area + b_area - c_area);
        }

        void cpu_nms(float *bboxes, PostprocessImageConfig postNetworkConfig)
        {
            int max_objects = postNetworkConfig.MAX_IMAGE_BOXES;
            float threshold = postNetworkConfig.nms_threshold_;
            int NUM_BOX_ELEMENT = postNetworkConfig.NUM_BOX_ELEMENT;
            int count = std::min((int)*bboxes, max_objects);

            for (int position = 0; position < count; position++)
            {
                // left, top, right, bottom, confidence, class, keepflag
                float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT; // +1是因为bboxes第一个值存储的是框的数量
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
        }
    }
}