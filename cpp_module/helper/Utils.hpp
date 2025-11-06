#pragma once

#include <filesystem>
#include <string>

#include "opencv2/opencv.hpp"
// ReSharper disable once CppUnusedIncludeDirective
#include "opencv2/core/ocl.hpp"

#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define RESET   "\033[0m"

#define LOG(...) std::cout << "[INFO] " << __VA_ARGS__ << RESET << std::endl;
#define LOG_ERR(...) std::cerr << RED << "[ERROR] " << __VA_ARGS__ << RESET << std::endl;
#define DEV_LOG(...) std::cerr << YELLOW << "[DEV INFO] " << __VA_ARGS__ << RESET << std::endl;

bool setUpEnv();
void errorHandler(const std::string&);
bool supportedWindowingSystem();
std::string GetTimestampString();
void handleWindow(std::string winName, const cv::Mat &frame, bool& quit);