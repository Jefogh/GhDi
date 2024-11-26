cmake_minimum_required(VERSION 3.10)
project(TrainedModel)

# تحديد معايير C++
set(CMAKE_CXX_STANDARD 17)

# إضافة المكتبات المطلوبة
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

# إضافة المسارات
include_directories(include)

# تعريف المكتبة
add_library(trained_model SHARED src/trained_model.cpp)

# ربط المكتبات
target_link_libraries(trained_model ${OpenCV_LIBS} openvino::runtime)
