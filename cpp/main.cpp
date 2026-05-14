#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "detector.hpp"
#include <sstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    if (argc == 4)
    {
        std::string video_path = argv[1];
        std::string model_path = argv[2];
        std::string output_path = argv[3];

        cv::Mat frame;

        cv::VideoCapture cap(video_path);

        if (!cap.isOpened())
        {
            std::cerr << "Cannot open video: " << video_path << std::endl;
            return 1;
        }

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        cv::VideoWriter writer;
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = 25.0;
        writer.open(output_path, codec, fps, cv::Size(width, height), true);

        TFLiteDetector detector(model_path);

        while (cap.read(frame))
        {
            if (frame.empty())
            {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }

            std::vector<Detection> result = detector.detect(frame);

            for (const Detection &detection : result)
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << detection.confidence;
                std::string conf = oss.str();

                cv::rectangle(frame, cv::Point(detection.bbox[0], detection.bbox[1]), cv::Point(detection.bbox[2], detection.bbox[3]), cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, detection.class_name + " " + conf, cv::Point(detection.bbox[0], detection.bbox[1] - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            }

            writer.write(frame);
        }
    }
    else
    {
        std::cout << "Usage: ./detect <video> <model> <output>" << std::endl;
    }
}