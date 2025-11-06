#include "Utils.hpp"
#include <stdexcept>

#include "classes/Yolo.hpp"
#ifdef  _WIN32
#include <ctime>
#endif
#ifdef __linux__
#include <stdlib.h>
#endif

void errorHandler(const std::string& msg) {
    if (msg.empty())
        LOG_ERR("Unknown Error occurred");
    LOG_ERR(msg);
    LOG("Press Enter to exit...");
    std::cin.get();
}

bool setUpEnv()
{
#ifdef _WIN32
    const std::filesystem::path opencv_kernel = std::filesystem::current_path() / "kernel_cache";

    if (!std::filesystem::exists(opencv_kernel))
    {
        if (!std::filesystem::create_directory(opencv_kernel))
        {
            LOG_ERR("Create Kernel Cache Failed");
            return false;
        }
    }

    if (_putenv_s("OPENCV_OCL4DNN_CONFIG_PATH", opencv_kernel.generic_string().c_str()) != 0)
    {
        LOG_ERR("SET Kernel Cache ENV Failed");
        return false;
    }
#endif

#ifdef __linux__
    if (setenv("NO_AT_BRIDGE", "1", 1) != 0)
    {
        LOG_ERR("SET NO_AT_BRIDGE ENV Failed");
        return false;
    }
#endif
    return true;
}


#ifdef  __linux__
bool supportedWindowingSystem()
{
    bool displayFound = false;
    if (const char *wayland_display = secure_getenv("WAYLAND_DISPLAY"); wayland_display != nullptr)
    {
        LOG("Wayland display detected");
        displayFound = true;
    }
    else if (const char *x11_display = secure_getenv("DISPLAY"); x11_display != nullptr)
    {
        LOG("X11 display detected");
        displayFound = true;
    }

    if (const char *xdg_session = secure_getenv("XDG_SESSION_TYPE"); xdg_session != nullptr)
    {
        LOG("Session type (XDG_SESSION_TYPE): " << xdg_session);
    }
    else
    {
        LOG("XDG_SESSION_TYPE environment variable not set.");
    }

    if (!displayFound)
    {
        LOG_ERR("No Wayland or X11 display environment variable found. The application may not be able to capture the screen.");
    }

    return displayFound;
}
#endif


std::string GetTimestampString()
{
    std::stringstream ss;
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    if (localtime_s(&tm,&t) == 0)
        ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return ss.str();
}


void handleWindow(std::string winName, const cv::Mat &frame, bool &quit)
{
    if(frame.empty())
        throw std::runtime_error("Frame is empty");

    if(winName.empty())
        winName = "Screenshot";

    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, 1280, 720);
     
    if (cv::getWindowProperty(winName, cv::WND_PROP_VISIBLE) >= 1)
    {
        cv::imshow(winName, frame);
    }

    if (const int key = cv::waitKey(20); key == 27)
    {
        quit = true;
    }
    
    if (cv::getWindowProperty(winName, cv::WND_PROP_VISIBLE) < 1)
    {
        quit = true;
    }
    
}