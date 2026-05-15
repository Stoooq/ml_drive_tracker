#include "detector.hpp"

int detect_frame(const char *model_path, uint8_t *frame_data, int width, int height, int channels, CDetection *out_detections, int max_detections)
{
    TFLiteDetector detector(model_path);

    cv::Mat frame(height, width, CV_8UC3, frame_data);

    std::vector<Detection> result = detector.detect(frame);

    for (int i = 0; i < std::min((int)result.size(), max_detections); i++)
    {
        std::memcpy(out_detections[i].bbox, result[i].bbox, sizeof out_detections[i].bbox);
        std::strncpy(out_detections[i].class_name, result[i].class_name.c_str(), sizeof out_detections[i].class_name);
        out_detections[i].confidence = result[i].confidence;
        out_detections[i].track_id = result[i].track_id.value_or(-1);
    }

    return result.size();
}