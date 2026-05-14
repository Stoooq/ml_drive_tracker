cmake_minimum_required(VERSION 3.16)
project(tflite_detector)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    tflite
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG v2.16.0
    GIT_SHALLOW TRUE 
    SOURCE_SUBDIR tensorflow/lite)
FetchContent_MakeAvailable(tflite)

find_package(OpenCV REQUIRED)

add_executable(tflite_detect main.cpp)

target_link_libraries(tflite_detect ${OpenCV_LIBS})
target_link_libraries(tflite_detect tensorflow-lite)
target_include_directories(tflite_detect PRIVATE ${tflite_SOURCE_DIR})