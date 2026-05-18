#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"

struct Detection
{
    int bbox[4];
    std::string class_name;
    float confidence;
    std::optional<int> track_id;
};

struct CDetection
{
    int bbox[4];
    char class_name[64];
    float confidence;
    int track_id;
};

class TFLiteDetector
{
private:
    std::string model_path;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> buffer;

public:
    TFLiteDetector(const std::string &model_path);
    std::vector<Detection> detect(cv::Mat frame, float confidence_threshold);
};

extern "C"
{
    int detect_frame(const char *model_path, uint8_t *frame_data, int width, int height, int channels, CDetection *out_detections, int max_detections, float confidence_threshold);
}