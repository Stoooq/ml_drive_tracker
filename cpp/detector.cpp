#include "detector.hpp"
#include <stdexcept>
#include <cmath>

static const std::unordered_map<int, std::string> COCO_NAMES = {
    {0, "person"},
    {1, "bicycle"},
    {2, "car"},
    {3, "motorcycle"},
    {4, "airplane"},
    {5, "bus"},
    {6, "train"},
    {7, "truck"},
    {8, "boat"},
    {9, "traffic light"},
    {10, "fire hydrant"},
    {11, "stop sign"},
    {12, "parking meter"},
    {13, "bench"},
    {14, "bird"},
    {15, "cat"},
    {16, "dog"},
    {17, "horse"},
    {18, "sheep"},
    {19, "cow"},
    {20, "elephant"},
    {21, "bear"},
    {22, "zebra"},
    {23, "giraffe"},
    {24, "backpack"},
    {25, "umbrella"},
    {26, "handbag"},
    {27, "tie"},
    {28, "suitcase"},
    {29, "frisbee"},
    {30, "skis"},
    {31, "snowboard"},
    {32, "sports ball"},
    {33, "kite"},
    {34, "baseball bat"},
    {35, "baseball glove"},
    {36, "skateboard"},
    {37, "surfboard"},
    {38, "tennis racket"},
    {39, "bottle"},
    {40, "wine glass"},
    {41, "cup"},
    {42, "fork"},
    {43, "knife"},
    {44, "spoon"},
    {45, "bowl"},
    {46, "banana"},
    {47, "apple"},
    {48, "sandwich"},
    {49, "orange"},
    {50, "broccoli"},
    {51, "carrot"},
    {52, "hot dog"},
    {53, "pizza"},
    {54, "donut"},
    {55, "cake"},
    {56, "chair"},
    {57, "couch"},
    {58, "potted plant"},
    {59, "bed"},
    {60, "dining table"},
    {61, "toilet"},
    {62, "tv"},
    {63, "laptop"},
    {64, "mouse"},
    {65, "remote"},
    {66, "keyboard"},
    {67, "cell phone"},
    {68, "microwave"},
    {69, "oven"},
    {70, "toaster"},
    {71, "sink"},
    {72, "refrigerator"},
    {73, "book"},
    {74, "clock"},
    {75, "vase"},
    {76, "scissors"},
    {77, "teddy bear"},
    {78, "hair drier"},
    {79, "toothbrush"},
};

TFLiteDetector::TFLiteDetector(const std::string &model_path)
    : model_path(model_path)
{
    buffer = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    if (buffer == nullptr)
    {
        std::cerr << "ERROR! buffer\n";
        throw std::runtime_error("opis błędu");
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*buffer, resolver)(&interpreter);

    if (interpreter == nullptr)
    {
        std::cerr << "ERROR! interpreter\n";
        throw std::runtime_error("opis błędu");
    }

    this->interpreter->AllocateTensors();
}

std::vector<Detection> TFLiteDetector::detect(cv::Mat frame)
{
    int input_index = interpreter->inputs()[0];
    TfLiteTensor *input_tensor = interpreter->tensor(input_index);

    int height = input_tensor->dims->data[1];
    int width = input_tensor->dims->data[2];

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(width, height));

    int8_t *input_data = interpreter->typed_tensor<int8_t>(input_index);

    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;

    for (int i = 0; i < resized.total() * resized.elemSize(); i++)
    {
        float data = resized.data[i];
        data /= 255.0f;
        data = (data / input_scale) + input_zero_point;
        input_data[i] = static_cast<int8_t>(std::round(data));
    }

    interpreter->Invoke();

    int output_index = interpreter->outputs()[0];
    TfLiteTensor *output_tensor = interpreter->tensor(output_index);

    int8_t *output_data = interpreter->typed_tensor<int8_t>(output_index);

    int num_detections = 8400;

    float scale = output_tensor->params.scale;
    int zero_point = output_tensor->params.zero_point;

    std::vector<Detection> result;
    int orig_height = frame.rows;
    int orig_width = frame.cols;
    float confidence_threshold = 0.4f;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_detections; i++)
    {
        float cx = (output_data[0 * num_detections + i] - zero_point) * scale;
        float cy = (output_data[1 * num_detections + i] - zero_point) * scale;
        float w = (output_data[2 * num_detections + i] - zero_point) * scale;
        float h = (output_data[3 * num_detections + i] - zero_point) * scale;

        float max_score = 0;
        int class_id = 0;
        for (int c = 0; c < 80; c++)
        {
            float score = (output_data[(4 + c) * num_detections + i] - zero_point) * scale;
            if (score > max_score)
            {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score > confidence_threshold)
        {
            float x1 = cx - (w / 2);
            float y1 = cy - (h / 2);
            float x2 = x1 + w;
            float y2 = y1 + h;

            int x1_px = x1 * orig_width;
            int y1_px = y1 * orig_height;
            int x2_px = x2 * orig_width;
            int y2_px = y2 * orig_height;

            boxes.push_back(cv::Rect{x1_px, y1_px, x2_px - x1_px, y2_px - y1_px});
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, 0.45f, indices);

    for (int &indice : indices)
    {
        cv::Rect box = boxes[indice];
        float confidence = confidences[indice];
        int class_id = class_ids[indice];

        result.push_back(Detection{
            {box.x, box.y, box.x + box.width, box.y + box.height},
            COCO_NAMES.at(class_id),
            confidence,
            std::nullopt,
        });
    }

    return result;
}