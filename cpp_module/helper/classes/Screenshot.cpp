#include "Screenshot.hpp"

Screenshot::Screenshot(const std::string &imagePath)
    : _path(imagePath)
{
    Init();
}

Screenshot::~Screenshot()
{
#if defined(_WIN32)
    DG::Cleanup(this->_ctx);
#endif
}

void Screenshot::Init()
{
#if defined(_WIN32)
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr))
    {
        throw std::runtime_error("Failed to initialize COM");
    }

    if (FAILED(DG::InitDesktopDuplication(this->_ctx)))
    {
        DG::Cleanup(this->_ctx);
        throw std::runtime_error("Failed to initialize Desktop Duplication");
    }
#elif defined(__linux__)

    const std::filesystem::path screenshotDir = std::filesystem::current_path() / this->_path;

    if (!std::filesystem::exists(screenshotDir))
    {
        if (!std::filesystem::create_directory(screenshotDir))
        {
            throw std::runtime_error("Unable to create screenshot directory");
        };
    }
#endif
}

void Screenshot::capture()
{
#if defined(_WIN32)
    HRESULT hr = DG::CaptureScreenshot(this->_ctx, this->_path, this->_screenshot);
    if (FAILED(hr))
    {
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_HUNG)
        {
            DG::Cleanup(this->_ctx);
            hr = DG::InitDesktopDuplication(this->_ctx);
            if (FAILED(hr))
            {
                throw std::runtime_error("Failed to reinitialize DirectX. Exiting");
            }
        }
        else if (hr == DXGI_ERROR_ACCESS_LOST)
        {
            if (this->_ctx.pDesktopDupl)
            {
                this->_ctx.pDesktopDupl->Release();
                this->_ctx.pDesktopDupl = nullptr;
            }

            hr = DG::InitDesktopDuplication(this->_ctx);
            if (FAILED(hr))
            {
                throw std::runtime_error("Failed to re-establish desktop duplication");
            }
        }
        else
        {
            throw std::runtime_error("Failed to capture screenshot");
        }
    }

#elif defined(__linux__)

    Display *display = XOpenDisplay(nullptr);
    if (!display)
    {
        throw std::runtime_error("Unable to open X display");
    }
    const Window root = DefaultRootWindow(display);

    XWindowAttributes gwa;
    XGetWindowAttributes(display, root, &gwa);

    auto image_deleter = [](XImage* img) { if (img) XDestroyImage(img); };
    std::unique_ptr<XImage, decltype(image_deleter)> img(
        XGetImage(display, root, 0, 0, gwa.width, gwa.height, AllPlanes, ZPixmap),
        image_deleter
    );

    if (!img)
    {
        throw std::runtime_error("Unable to get Image");
    }
    std::string fileName = "screenshot_" + GetTimestampString() + ".png";
    const std::filesystem::path fullPath = this->_path + "/" + fileName;

    // Create cv::Mat from XImage data (assuming 32bpp)
    const cv::Mat mat(gwa.height, gwa.width, CV_8UC4, img->data);

    // Convert BGRA to BGR if needed
    cv::cvtColor(mat, this->_screenshot, cv::COLOR_BGRA2BGR);

    // Save to file
    if (!cv::imwrite(fullPath.generic_string(), this->_screenshot))
    {
        throw std::runtime_error("Unable to save screenshot");
    }
    LOG("Saved to: " << fullPath.generic_string());
#endif
}

const cv::Mat &Screenshot::getImage() const
{
    return this->_screenshot;
}
