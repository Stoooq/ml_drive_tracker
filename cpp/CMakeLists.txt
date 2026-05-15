cmake_minimum_required(VERSION 3.16)
project(tflite_detector)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
set(ABSL_RANDOM_RANDEN_COPTS "" CACHE STRING "" FORCE)

include(FetchContent)

FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20240722.0
    GIT_SHALLOW TRUE
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(abseil-cpp)

FetchContent_Declare(
    tflite
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG v2.19.1
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR tensorflow/lite)
FetchContent_MakeAvailable(tflite)

find_package(OpenCV REQUIRED)

add_executable(tflite_detect main.cpp detector.cpp)

target_link_libraries(tflite_detect ${OpenCV_LIBS})
target_link_libraries(tflite_detect tensorflow-lite)
target_include_directories(tflite_detect PRIVATE ${tflite_SOURCE_DIR})

add_library(tflite_detect_lib SHARED detector.cpp detector_lib.cpp)

target_link_libraries(tflite_detect_lib ${OpenCV_LIBS})
target_link_libraries(tflite_detect_lib tensorflow-lite)
target_include_directories(tflite_detect_lib PRIVATE ${tflite_SOURCE_DIR})