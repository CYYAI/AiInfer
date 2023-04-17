#ifndef _POST_PROCESS_HPP_CPP_
#define _POST_PROCESS_HPP_CPP_
#include <algorithm>
#include "common/model_info.hpp"

namespace ai
{
    namespace postprocess
    {
        using namespace ai::modelInfo;
        void yolov8_decode_network_out_trans(float *predict, int num_bboxes, int num_classes, int output_cdim, float confidence_threshold,
                                             float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT,
                                             const float scale, const int fill_width, const int fill_height);
        float box_iou(float aleft, float atop, float aright, float abottom,
                      float bleft, float btop, float bright, float bbottom);
        void cpu_nms(float *bboxes, PostprocessImageConfig postNetworkConfig);
    }
}
#endif // _POST_PROCESS_HPP_CPP_