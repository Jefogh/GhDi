#include "trained_model.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    TrainedModel model;

    // تحميل صورة للاختبار
    cv::Mat img = cv::imread("test_image.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    auto [num1, operation, num2] = model.predict(img);

    std::cout << "Prediction: " << num1 << " " << operation << " " << num2 << std::endl;
    return 0;
}
