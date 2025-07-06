#pragma once

#include <vector>
#include <chrono>
#include <thread>
#include <iostream>
#include <wrl/client.h>
#include <thread>
#include <filesystem>
#include <string>
#include <sstream>
#include <iomanip>

// Windows and DirectX headers
#include <dxgi1_2.h>
#include <d3d11.h>
#include <windows.h>
#include <wincodec.h>

// OpenCV includes
#include <opencv2/opencv.hpp>

using Microsoft::WRL::ComPtr;

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "windowscodecs.lib")

namespace DG
{
// --- Helper for HRESULT error checking ---
// This macro simplifies error handling for DirectX functions.
// If an HRESULT indicates failure, it prints an error message and exits.
#define CHECK_HR(hr, msg)                                                                            \
    if (FAILED(hr))                                                                                  \
    {                                                                                                \
        std::cerr << "Error: " << msg << ". HRESULT: 0x" << std::hex << hr << std::dec << std::endl; \
        return hr;                                                                                   \
    }

    template <class T>
    void SafeRelease(T **ppT)
    {
        if (*ppT)
        {
            (*ppT)->Release();
            *ppT = nullptr;
        }
    }

    bool GetScreenPixelsDXGI(
        IDXGIOutputDuplication *pDuplication,
        ID3D11Device *pDevice,
        ID3D11DeviceContext *pImmediateContext,
        int &width,
        int &height,
        std::vector<BYTE> &pixel_data_out);

    // Helper struct to hold DXGI/DirectX objects
    struct DXGIContext
    {
        ID3D11Device *pDevice = nullptr;
        ID3D11DeviceContext *pImmediateContext = nullptr;
        IDXGIFactory1 *pFactory = nullptr;
        IDXGIAdapter1 *pAdapter = nullptr;
        IDXGIOutput1 *pOutput1 = nullptr;
        IDXGIOutputDuplication *pDesktopDupl = nullptr;
        IWICImagingFactory *pWICFactory = nullptr;
    };

    void CleanupDXGI(DXGIContext &ctx);
    bool InitializeDXGI(DXGIContext &ctx);
    std::string GetTimestampString();
    HRESULT InitDesktopDuplication(DXGIContext &ctx);
    void Cleanup(DXGIContext &ctx);
    HRESULT SavePixelsToPng(DXGIContext &ctx, const std::string &imageDirectory, const BYTE *pixels, UINT width, UINT height, UINT pitch, cv::Mat &out_cv_image);
    bool IsScreenBlack(const BYTE *pixels, UINT width, UINT height, UINT pitch);
    HRESULT CaptureScreenshot(DXGIContext &ctx, const std::string &outputPath, cv::Mat &out_cv_image);
}