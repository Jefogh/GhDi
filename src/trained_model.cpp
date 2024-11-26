#include "trained_model.h"
#include <iostream>
#include <chrono>

TrainedModel::TrainedModel() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // تحميل نموذج OpenVINO
    std::string model_path = "C:/Users/ccl/Desktop/trained_model.xml";
    ov::Core core;
    auto model = core.read_model(model_path);
    auto compiled_model = core.compile_model(model, "GPU");
    auto infer_request = compiled_model.create_infer_request();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Model loaded in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << " ms" << std::endl;
}

// تعريف لدالة التوقع
std::tuple<int, std::string, int> TrainedModel::predict(const cv::Mat& img) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // تغيير حجم الصورة
    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(160, 90));

    auto resize_time = std::chrono::high_resolution_clock::now();
    std::cout << "Image resizing took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(resize_time - start_time).count()
              << " ms" << std::endl;

    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    cv::Mat chw_image(3, cv::Size(160, 90), CV_32F);
    std::vector<cv::Mat> channels(3);
    cv::split(resized_image, channels);
    for (auto& channel : channels) {
        channel -= 0.5f;
        channel /= 0.5f;
    }
    cv::merge(channels, chw_image);

    // إرسال الصورة إلى النموذج
    ov::Tensor input_tensor = ov::Tensor(ov::element::f32, {1, 3, 90, 160}, chw_image.data);

    // استدعاء التوقع
    ov::InferRequest infer_request;
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto outputs = infer_request.get_output_tensor();
    float* output_data = outputs.data<float>();
    int output_size = outputs.get_size();

    // تقسيم المخرجات
    int num1_predicted = std::max_element(output_data, output_data + 10) - output_data;
    int operation_predicted = std::max_element(output_data + 10, output_data + 13) - (output_data + 10);
    int num2_predicted = std::max_element(output_data + 13, output_data + output_size) - (output_data + 13);

    std::string operation_map[3] = {"+", "-", "×"};
    std::string predicted_operation = operation_map[operation_predicted];

    return std::make_tuple(num1_predicted, predicted_operation, num2_predicted);
}
