#pragma once

#include "utils.hpp"

#if defined(_WIN32)
#include "dxdiag.hpp"
#elif defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

class Screenshot
{
private:
#if defined(_WIN32)
    DG::DXGIContext _ctx;
#endif
    std::string _path;
    cv::Mat _screenshot;
    void Init();

public:
    Screenshot(const std::string &imagePath);
    ~Screenshot();
    void capture();
    const cv::Mat &getImage() const;
};
