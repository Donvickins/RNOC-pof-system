#include "utils.hpp"
#include <stdexcept>

#ifdef _WIN32
#include <cstdlib>
#else
#include <stdlib.h>
#endif

bool setUpEnv()
{
#ifdef _WIN32
    std::filesystem::path opencv_kernel = std::filesystem::current_path() / "Kernel_cache";

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

#if (__linux__)
    if (setenv("NO_AT_BRIDGE", "1", 1) != 0)
    {
        LOG_ERR("SET NO_AT_BRIDGE ENV Failed");
        return false;
    }
#endif
    return true;
}



bool supportedWindowingSystem()
{
    bool displayFound = false;
    if (const char *wayland_display = getenv("WAYLAND_DISPLAY"); wayland_display != nullptr)
    {
        LOG("Wayland display detected");
        displayFound = true;
    }
    else if (const char *x11_display = getenv("DISPLAY"); x11_display != nullptr)
    {
        LOG("X11 display detected");
        displayFound = true;
    }

    if (const char *xdg_session = getenv("XDG_SESSION_TYPE"); xdg_session != nullptr)
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

std::string GetTimestampString()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}


void handleWindow(std::string winname, cv::Mat &frame, bool &quit)
{
    if(frame.empty())
        throw std::runtime_error("Frame is empty");

    if(winname.empty())
        winname = "Screenshot";

    cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::resizeWindow(winname, 1280, 720);

     
    if (cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE) >= 1)
    {
        cv::imshow(winname, frame);
    }

    int key = cv::waitKey(10);
    if (key == 27)
    {
        quit = true;
    }
    
    if (cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE) < 1)
    {
        quit = true;
    }
    
}