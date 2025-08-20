#pragma once

#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

#include "utils.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/core/utils/logger.hpp"

const float CONFIDENCE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.4f;
const int YOLO_INPUT_WIDTH = 640;
const int YOLO_INPUT_HEIGHT = 640;

bool processFrameWithYOLO(cv::Mat &frame, cv::dnn::Net &net, const std::vector<std::string> &class_names_list);
bool loadClassNames(const std::string &path, std::vector<std::string> &class_names_out);
bool setupYoloNetwork(cv::dnn::Net &net, const std::string &model_path, const std::string &class_names_path, std::vector<std::string> &out_class_names_vec, HARDWARE_INFO &hw_info);