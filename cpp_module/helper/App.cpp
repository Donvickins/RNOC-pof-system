#include "App.hpp"

#ifndef _WIN32
#include "dxdiag.hpp"
#endif

bool initYolo(YOLO& model) {
	model.HardwareSummary();
	try
	{
		model.Init();
	}
	catch (const cv::Exception &e)
	{
		errorHandler(std::format("Failed to load model: {}", e.msg));
		return false;
	}
	catch (const std::exception &e)
	{
		errorHandler(std::format("Failed to load model: {}", e.what()));
		return false;
	}
	return true;
}