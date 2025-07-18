#pragma once

#include "utils.hpp"
#include <memory>

#if defined(_WIN32)
#include "dxdiag.hpp"
#elif defined(__linux__)
// Forward declare Display to avoid including X11 headers in a public header.
struct _XDisplay;
using Display = struct _XDisplay;
#endif

class Screenshot
{
private:
#if defined(_WIN32)
    DG::DXGIContext _ctx;
#elif defined(__linux__)
    struct DisplayDeleter
    {
        void operator()(Display *d) const;
    };
    std::unique_ptr<Display, DisplayDeleter> _display;
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
