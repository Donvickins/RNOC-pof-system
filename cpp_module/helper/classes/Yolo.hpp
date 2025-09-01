#pragma once

#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

#include "utils.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/core/utils/logger.hpp"

struct HINFO
{
    bool has_cuda = false;
    bool has_opencl = false;
    bool has_amd = false;
    bool has_intel = false;
    bool has_nvidia = false;
    std::string gpu_name;
    std::string gpu_vendor;
};

class YOLO
{
private:
    const float CONFIDENCE_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.4f;
    const int YOLO_INPUT_WIDTH = 640;
    const int YOLO_INPUT_HEIGHT = 640;
    cv::dnn::Net model;
    const std::string MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolov8l.onnx").generic_string();
    std::vector<std::string> class_names;
    const std::string class_names_path = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();
    HINFO hw_info;
    void LoadClassNames();
    void SetupYoloNetwork();
    void CheckGPU();

public:
    YOLO();
    void Init();
    void HardwareSummary();
    void ProcessFrame(cv::Mat &frame);
};