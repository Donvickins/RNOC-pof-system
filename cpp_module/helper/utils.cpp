#include "utils.hpp"

#ifdef _WIN32
#include <cstdlib>
#else
#include <stdlib.h>
#endif

bool setUpEnv()
{
#ifdef _WIN32
    std::filesystem::path opencv_kernel = std::filesystem::current_path() / "kernel_cache";

    if (!std::filesystem::exists(opencv_kernel))
    {
        if (!std::filesystem::create_directory(opencv_kernel))
        {
            LOG_ERR("Create Kernel Cache Failed");
            return false;
        }

        if (_putenv_s("OPENCV_OCL4DNN_CONFIG_PATH", opencv_kernel.generic_string().c_str()) != 0)
        {
            LOG_ERR("SET Kernel Cache ENV Failed");
            return false;
        }
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

void checkGPU(HARDWARE_INFO &hw_info)
{
    // Check CUDA (NVIDIA)
    try
    {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0)
        {
            hw_info.has_cuda = true;
            hw_info.has_nvidia = true;
        }
    }
    catch (const cv::Exception &e)
    {
        LOG_ERR("CUDA check failed: " << e.what());
    }

    // Check OpenCL (AMD, Intel, NVIDIA)
    if (cv::ocl::haveOpenCL())
    {
        cv::ocl::Context context;
        if (context.create(cv::ocl::Device::TYPE_ALL))
        {
            hw_info.has_opencl = true;
            cv::ocl::Device device = context.device(0);
            hw_info.gpu_name = device.name();
            hw_info.gpu_vendor = device.vendorName();

            // Detect vendor
            if (hw_info.gpu_vendor.find("AMD") != std::string::npos)
            {
                hw_info.has_amd = true;
            }
            else if (hw_info.gpu_vendor.find("Intel") != std::string::npos)
            {
                hw_info.has_intel = true;
            }
            else if (hw_info.gpu_vendor.find("NVIDIA") != std::string::npos)
            {
                hw_info.has_nvidia = true;
            }
        }
    }
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

void hardwareSummary(HARDWARE_INFO &hw_info)
{
    LOG("Hardware Detection Summary");
    if (!hw_info.has_cuda && !hw_info.has_opencl)
    {
        LOG("No GPU acceleration detected. Using CPU backend.");
        return;
    }

    LOG("GPU Vendor: " << (hw_info.gpu_vendor.empty() ? "N/A" : hw_info.gpu_vendor));
    LOG("GPU Name:   " << (hw_info.gpu_name.empty() ? "N/A" : hw_info.gpu_name));

    if (hw_info.has_cuda)
    {
        LOG("Backend:    CUDA enabled. (Optimal performance)");
    }
    else if (hw_info.has_opencl)
    {
        LOG("Backend:    OpenCL enabled.");
        if (hw_info.has_nvidia)
        {
            LOG("Note:       For best performance on NVIDIA GPUs, please install the CUDA Toolkit.");
        }
    }
}