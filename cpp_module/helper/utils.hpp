#pragma once

#include <filesystem>
#include <iostream>
#include <optional>

#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

#define LOG(...) std::cout << "[INFO] " << __VA_ARGS__ << std::endl;
#define LOG_ERR(...) std::cerr << "[ERROR] " << __VA_ARGS__ << std::endl;

bool setUpEnv();
bool supportedWindowingSystem();
std::string GetTimestampString();
void handleWindow(std::string winname, cv::Mat &frame, bool& quit);