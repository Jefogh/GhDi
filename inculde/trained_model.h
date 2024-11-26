#ifndef TRAINED_MODEL_H
#define TRAINED_MODEL_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <tuple>
#include <string>

class TrainedModel {
public:
    TrainedModel();
    std::tuple<int, std::string, int> predict(const cv::Mat& img);
};

#endif
