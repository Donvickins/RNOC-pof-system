#include "dxdiag.hpp"
#include <opencv2/opencv.hpp>

namespace DG
{

    void enableANSIColors() {
        const HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }

    bool GetScreenPixelsDXGI(
        IDXGIOutputDuplication *pDuplication,
        ID3D11Device *pDevice,
        ID3D11DeviceContext *pImmediateContext,
        int &width,
        int &height,
        std::vector<BYTE> &pixel_data_out)
    {
        static const int MAX_RETRIES = 3;
        static const int RETRY_DELAY_MS = 10;

        for (int retry = 0; retry < MAX_RETRIES; retry++)
        {
            IDXGIResource *pDesktopResource = nullptr;
            DXGI_OUTDUPL_FRAME_INFO frameInfo;

            // Try to acquire a new frame with a small timeout
            HRESULT hr = pDuplication->AcquireNextFrame(100, &frameInfo, &pDesktopResource);

            if (hr == DXGI_ERROR_WAIT_TIMEOUT)
            {
                // No new frame available yet, this is normal
                SafeRelease(&pDesktopResource);
                if (retry < MAX_RETRIES - 1)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                    continue;
                }
                return false;
            }

            if (FAILED(hr) || !pDesktopResource)
            {
                std::cerr << "Failed to acquire next frame. HR: " << std::hex << hr << std::endl;
                if (hr == DXGI_ERROR_ACCESS_LOST)
                {
                    std::cerr << "Access to desktop duplication was lost (e.g. mode change, fullscreen app). Re-initialization needed." << std::endl;
                }
                SafeRelease(&pDesktopResource);
                if (retry < MAX_RETRIES - 1)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                    continue;
                }
                return false;
            }

            // Query the texture interface from the resource
            ID3D11Texture2D *pAcquiredDesktopImage = nullptr;
            hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&pAcquiredDesktopImage));
            SafeRelease(&pDesktopResource); // Release the IDXGIResource, we have the texture now or failed

            if (FAILED(hr) || !pAcquiredDesktopImage)
            {
                std::cerr << "Failed to query ID3D11Texture2D from IDXGIResource. HR: " << std::hex << hr << std::endl;
                if (retry < MAX_RETRIES - 1)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                    continue;
                }
                return false;
            }

            D3D11_TEXTURE2D_DESC desc;
            pAcquiredDesktopImage->GetDesc(&desc);

            // Create a staging texture to copy the desktop image to
            ID3D11Texture2D *pStagingTexture = nullptr;
            D3D11_TEXTURE2D_DESC stagingDesc = desc;
            stagingDesc.Usage = D3D11_USAGE_STAGING;
            stagingDesc.BindFlags = 0;
            stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            stagingDesc.MiscFlags = 0;

            hr = pDevice->CreateTexture2D(&stagingDesc, nullptr, &pStagingTexture);
            if (FAILED(hr) || !pStagingTexture)
            {
                std::cerr << "Failed to create staging texture. HR: " << std::hex << hr << std::endl;
                SafeRelease(&pAcquiredDesktopImage);
                if (retry < MAX_RETRIES - 1)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                    continue;
                }
                return false;
            }

            // Copy the desktop image to the staging texture
            pImmediateContext->CopyResource(pStagingTexture, pAcquiredDesktopImage);
            SafeRelease(&pAcquiredDesktopImage);

            // Map the staging texture to access the pixel data
            D3D11_MAPPED_SUBRESOURCE mappedResource;
            hr = pImmediateContext->Map(pStagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
            if (FAILED(hr))
            {
                std::cerr << "Failed to map staging texture. HR: " << std::hex << hr << std::endl;
                SafeRelease(&pStagingTexture);
                if (retry < MAX_RETRIES - 1)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                    continue;
                }
                return false;
            }

            // Update the output dimensions
            width = desc.Width;
            height = desc.Height;

            // Calculate the total size needed for the pixel data
            size_t totalSize = width * height * 4; // 4 bytes per pixel (BGRA)
            pixel_data_out.resize(totalSize);

            // Copy the pixel data row by row
            BYTE *pSrcData = static_cast<BYTE *>(mappedResource.pData);
            BYTE *pDstData = pixel_data_out.data();
            for (int row = 0; row < height; row++)
            {
                memcpy(pDstData + row * width * 4,
                       pSrcData + row * mappedResource.RowPitch,
                       width * 4);
            }

            // Unmap the staging texture
            pImmediateContext->Unmap(pStagingTexture, 0);
            SafeRelease(&pStagingTexture);

            // Release the frame
            hr = pDuplication->ReleaseFrame();
            if (FAILED(hr))
            {
                std::cerr << "Failed to release frame. HR: " << std::hex << hr << std::endl;
                if (retry < MAX_RETRIES - 1)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                    continue;
                }
                return false;
            }

            return true;
        }

        return false;
    }

    void CleanupDXGI(DXGIContext &ctx)
    {
        SafeRelease(&ctx.pDesktopDupl);
        SafeRelease(&ctx.pOutput1);
        SafeRelease(&ctx.pImmediateContext);
        SafeRelease(&ctx.pDevice);
        SafeRelease(&ctx.pAdapter);
        SafeRelease(&ctx.pFactory);
    }

    // Initialize DXGI/DirectX and duplication objects. Returns true on success.
    bool InitializeDXGI(DXGIContext &ctx)
    {
        HRESULT hr;

        // Create DXGI Factory
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&ctx.pFactory));
        if (FAILED(hr))
        {
            std::cerr << "Failed to create DXGI Factory. HR: " << std::hex << hr << std::endl;
            return false;
        }

        // Feature levels to try for D3D11 device creation
        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_3,
            D3D_FEATURE_LEVEL_9_1,
        };
        D3D_FEATURE_LEVEL featureLevel;

        UINT adapterIndex = 0;
        ComPtr<IDXGIAdapter1> pAdapter; // Use ComPtr for automatic release

        while (ctx.pFactory->EnumAdapters1(adapterIndex, &pAdapter) != DXGI_ERROR_NOT_FOUND)
        {
            DXGI_ADAPTER_DESC1 adapterDesc;
            pAdapter->GetDesc1(&adapterDesc);

            // Skip software adapters (WARP) unless specifically desired
            if (adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                adapterIndex++;
                continue;
            }

            // std::wcout << L"Found Adapter: " << adapterDesc.Description << std::endl;

            UINT outputIndex = 0;
            ComPtr<IDXGIOutput> pOutput;

            while (pAdapter->EnumOutputs(outputIndex, &pOutput) != DXGI_ERROR_NOT_FOUND)
            {
                DXGI_OUTPUT_DESC outputDesc;
                pOutput->GetDesc(&outputDesc);

                if (outputDesc.AttachedToDesktop)
                {
                    // std::wcout << L"  Found Output: " << outputDesc.DeviceName << L" (Attached to Desktop)" << std::endl;

                    // Try to create D3D11 device on this specific adapter
                    hr = D3D11CreateDevice(
                        pAdapter.Get(),          // Use the current adapter
                        D3D_DRIVER_TYPE_UNKNOWN, // Driver type is UNKNOWN when an adapter is explicitly specified
                        nullptr,
                        0,
                        featureLevels,
                        ARRAYSIZE(featureLevels),
                        D3D11_SDK_VERSION,
                        &ctx.pDevice,
                        &featureLevel,
                        &ctx.pImmediateContext);

                    if (SUCCEEDED(hr))
                    {
                        // std::cout << "  D3D11 Device created successfully on this adapter." << std::endl;

                        // Query for IDXGIOutput1 interface from the *same* output
                        hr = pOutput->QueryInterface(__uuidof(IDXGIOutput1), reinterpret_cast<void **>(&ctx.pOutput1));
                        if (SUCCEEDED(hr))
                        {
                            // std::cout << "  IDXGIOutput1 queried successfully." << std::endl;

                            // Attempt to duplicate the output
                            hr = ctx.pOutput1->DuplicateOutput(ctx.pDevice, &ctx.pDesktopDupl);
                            if (SUCCEEDED(hr))
                            {
                                // std::cout << "Desktop Duplication initialized successfully on output: " << outputIndex << " of adapter: " << adapterIndex << std::endl;
                                // We found a working setup, store references and return true
                                ctx.pAdapter = pAdapter.Detach(); // Detach to keep the reference
                                // pOutput.Detach() is not needed as ctx.pOutput1 is derived from it
                                return true;
                            }
                            else
                            {
                                LOG_ERR("Failed to create duplicate output");
                                if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE)
                                {
                                    LOG_ERR("Desktop Duplication is not available. Max number of applications using it already reached or a fullscreen application is running.");
                                }
                                else if (hr == E_ACCESSDENIED)
                                {
                                    std::cerr << "  Access denied. Possibly due to protected content or system settings (e.g., Secure Desktop)." << std::endl;
                                }
                                // Clean up the device and context if duplication failed
                                SafeRelease(&ctx.pImmediateContext);
                                SafeRelease(&ctx.pDevice);
                                SafeRelease(&ctx.pOutput1);
                            }
                        }
                        else
                        {
                            LOG_ERR("Failed to query IDXGIOutput1 from output. HR: 0x" << std::hex << hr << std::dec);
                            SafeRelease(&ctx.pImmediateContext);
                            SafeRelease(&ctx.pDevice);
                        }
                    }
                    else
                    {
                        LOG_ERR("Failed to create D3D11 device on this adapter. HR: 0x" << std::hex << hr << std::dec);
                    }
                }
                outputIndex++;
                pOutput.Reset(); // Release current output before enumerating next
            }
            adapterIndex++;
            pAdapter.Reset(); // Release current adapter before enumerating next
        }

        LOG_ERR("Failed to initialize DXGI on any suitable adapter/output combination.");
        SafeRelease(&ctx.pFactory); // Ensure factory is released if nothing was found
        return false;
    }

    // Helper function to get formatted timestamp
    std::string GetTimestampString()
    {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        return ss.str();
    }

    HRESULT InitDesktopDuplication(DXGIContext &ctx)
    {
        HRESULT hr = S_OK;

        // 1. Create a DirectX 11 Device
        // This device represents the display adapter and is used for rendering and computation.
        hr = D3D11CreateDevice(
            nullptr,                  // Adapter: Use default adapter
            D3D_DRIVER_TYPE_HARDWARE, // Driver Type: Use hardware acceleration
            nullptr,                  // Software Rasterizer: Not used for hardware type
            0,                        // Flags: No special flags
            nullptr,                  // Feature Levels: Array of feature levels to try (nullptr uses default)
            0,                        // Number of Feature Levels: 0 for default
            D3D11_SDK_VERSION,        // SDK Version
            &ctx.pDevice,             // Output: Pointer to the created device
            nullptr,                  // Output: Pointer to the determined feature level
            &ctx.pImmediateContext    // Output: Pointer to the created device context
        );
        CHECK_HR(hr, "Failed to create D3D11 device");

        // 2. Get DXGI Factory
        // DXGI (DirectX Graphics Infrastructure) is used for enumerating graphics adapters and outputs.
        IDXGIDevice *DxgiDevice = nullptr;
        hr = ctx.pDevice->QueryInterface(__uuidof(IDXGIDevice), (void **)&DxgiDevice);
        CHECK_HR(hr, "Failed to query IDXGIDevice from D3D11 device");

        IDXGIAdapter *DxgiAdapter = nullptr;
        hr = DxgiDevice->GetAdapter(&DxgiAdapter);
        DxgiDevice->Release(); // Release the intermediate DxgiDevice
        CHECK_HR(hr, "Failed to get IDXGIAdapter from IDXGIDevice");

        IDXGIFactory1 *DxgiFactory = nullptr;
        hr = DxgiAdapter->GetParent(__uuidof(IDXGIFactory1), (void **)&DxgiFactory);
        DxgiAdapter->Release(); // Release the intermediate DxgiAdapter
        CHECK_HR(hr, "Failed to get IDXGIFactory1 from IDXGIAdapter");

        // 3. Get the primary display output
        // We'll target the first display output (index 0).
        IDXGIOutput *DxgiOutput = nullptr;
        hr = DxgiFactory->EnumAdapters(0, &DxgiAdapter);
        CHECK_HR(hr, "Failed to enumerate DXGI adapter");

        hr = DxgiAdapter->EnumOutputs(0, &DxgiOutput); // Enumerate the first output
        DxgiFactory->Release();                        // Release the intermediate DxgiFactory
        DxgiAdapter->Release();                        // Release the adapter
        CHECK_HR(hr, "Failed to enumerate DXGI output");

        // 4. Get the Desktop Duplication Interface
        // This is the core API for capturing the desktop image.
        IDXGIOutput1 *DxgiOutput1 = nullptr;
        hr = DxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void **)&DxgiOutput1);
        DxgiOutput->Release(); // Release the intermediate DxgiOutput
        CHECK_HR(hr, "Failed to query IDXGIOutput1 from IDXGIOutput");

        // Create the desktop duplication interface for the output
        hr = DxgiOutput1->DuplicateOutput(ctx.pDevice, &ctx.pDesktopDupl);
        DxgiOutput1->Release(); // Release the intermediate DxgiOutput1
        CHECK_HR(hr, "Failed to duplicate desktop output. Make sure you have Windows 8 or later and proper permissions.");

        // 5. Initialize WIC Factory
        // WIC is used for encoding and decoding image formats (like PNG, BMP).
        hr = CoCreateInstance(
            CLSID_WICImagingFactory,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&ctx.pWICFactory));
        CHECK_HR(hr, "Failed to create WIC Imaging Factory. Ensure COM is initialized (CoInitializeEx).");

        LOG("DirectX and Desktop Duplication initialized successfully");
        return hr;
    }

    void Cleanup(DXGIContext &ctx)
    {
        if (ctx.pDesktopDupl)
        {
            ctx.pDesktopDupl->Release();
            ctx.pDesktopDupl = nullptr;
        }
        if (ctx.pDevice)
        {
            ctx.pImmediateContext->Release();
            ctx.pImmediateContext = nullptr;
        }
        if (ctx.pDevice)
        {
            ctx.pDevice->Release();
            ctx.pDevice = nullptr;
        }
        if (ctx.pWICFactory)
        {
            ctx.pWICFactory->Release();
            ctx.pWICFactory = nullptr;
        }
        CoUninitialize(); // Uninitialize COM
        std::cout << "DirectX and WIC components cleaned up." << std::endl;
    }

    bool IsScreenBlack(const BYTE *pixels, UINT width, UINT height, UINT pitch)
    {
        if (!pixels)
            return true; // Treat null data as black/invalid

        // Check a sample of pixels to quickly determine if it's black.
        // For a more robust check, you might iterate through more pixels or calculate an average.
        const int sample_size = 100; // Check 100 pixels
        int non_black_pixels = 0;

        for (int i = 0; i < sample_size; ++i)
        {
            // Randomly select a pixel to check
            UINT x = rand() % width;
            UINT y = rand() % height;

            // Calculate byte offset for BGRA format (4 bytes per pixel)
            // Pitch is the row stride in bytes
            const BYTE *pixel_data = pixels + (y * pitch) + (x * 4);

            // Check if any of the color components (B, G, R) are non-zero.
            // Alpha (A) is usually 255 for opaque, but we care about color.
            if (pixel_data[0] != 0 || pixel_data[1] != 0 || pixel_data[2] != 0)
            {
                non_black_pixels++;
            }
        }

        // If a significant portion of sampled pixels are non-black, it's not black.
        return non_black_pixels < (sample_size / 10); // If less than 10% are non-black, consider it black
    }

    HRESULT SavePixelsToPng(DXGIContext &ctx, const std::string &imageDirectory, const BYTE *pixels, UINT width, UINT height, UINT pitch, cv::Mat &out_cv_image)
    {
        HRESULT hr = S_OK;
        IWICStream *pStream = nullptr;
        IWICBitmapEncoder *pEncoder = nullptr;
        IWICBitmapFrameEncode *pFrameEncode = nullptr;
        WICPixelFormatGUID pixelFormat = GUID_WICPixelFormat32bppBGRA; // Our captured format

        hr = ctx.pWICFactory->CreateStream(&pStream);
        CHECK_HR(hr, "Failed to create WIC stream");

        try
        {
            if (!std::filesystem::exists(imageDirectory))
            {
                std::filesystem::create_directories(imageDirectory);
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            std::cerr << "Filesystem error creating directory: " << e.what() << std::endl;
            return E_FAIL; // Return a general failure HRESULT
        }

        std::string filename = "screenshot_" + GetTimestampString() + ".png";
        std::filesystem::path fullPath = std::filesystem::path(imageDirectory) / filename;

        // Initialize stream to write to the specified file
        hr = pStream->InitializeFromFilename(fullPath.wstring().c_str(), GENERIC_WRITE);
        CHECK_HR(hr, "Failed to initialize WIC stream from filename");

        // Create a PNG encoder
        hr = ctx.pWICFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &pEncoder);
        CHECK_HR(hr, "Failed to create PNG encoder");

        hr = pEncoder->Initialize(pStream, WICBitmapEncoderNoCache);
        CHECK_HR(hr, "Failed to initialize PNG encoder");

        hr = pEncoder->CreateNewFrame(&pFrameEncode, nullptr);
        CHECK_HR(hr, "Failed to create new frame for encoder");

        hr = pFrameEncode->Initialize(nullptr);
        CHECK_HR(hr, "Failed to initialize frame encoder");

        // Set frame dimensions
        hr = pFrameEncode->SetSize(width, height);
        CHECK_HR(hr, "Failed to set frame size");

        // Set pixel format (BGRA is common for screen capture)
        hr = pFrameEncode->SetPixelFormat(&pixelFormat);
        CHECK_HR(hr, "Failed to set pixel format");

        // Check if the pixel format was successfully set
        // This is important because WIC might choose a different format if the requested one is not supported
        // For Desktop Duplication, BGRA is usually fine.
        if (pixelFormat != GUID_WICPixelFormat32bppBGRA)
        {
            std::cerr << "Warning: Pixel format conversion occurred, this might affect performance or quality." << std::endl;
            // You might need to convert pixels if the format changed. For simplicity, we assume BGRA.
        }

        // Write the actual pixel data
        hr = pFrameEncode->WritePixels(height, pitch, pitch * height, const_cast<BYTE *>(pixels));
        CHECK_HR(hr, "Failed to write pixels to frame");

        hr = pFrameEncode->Commit();
        CHECK_HR(hr, "Failed to commit frame");

        hr = pEncoder->Commit();
        CHECK_HR(hr, "Failed to commit encoder");

        LOG("Screenshot saved to: " << filename);

        // Release WIC interfaces
        if (pFrameEncode)
            pFrameEncode->Release();
        if (pEncoder)
            pEncoder->Release();
        if (pStream)
            pStream->Release();

        out_cv_image = cv::Mat(height, width, CV_8UC4, const_cast<BYTE *>(pixels), pitch);
        cv::cvtColor(out_cv_image, out_cv_image, cv::COLOR_BGRA2BGR);

        return hr;
    }

    HRESULT CaptureScreenshot(DXGIContext &ctx, const std::string &outputPath, cv::Mat &out_cv_image)
    {
        HRESULT hr = S_OK;

        IDXGIResource *DesktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO FrameInfo;

        // 1. Acquire next frame from the desktop duplication interface
        // This is a blocking call that waits for a new desktop image.
        // Timeout of 100ms: if no new frame in 100ms, it might be a static screen.
        hr = ctx.pDesktopDupl->AcquireNextFrame(100, &FrameInfo, &DesktopResource);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            // No new frame available within the timeout. This can happen if the screen is static.
            // It's not necessarily an error, just means nothing changed.
            return S_OK; // Return S_OK as it's not a critical failure for this loop
        }
        CHECK_HR(hr, "Failed to acquire next frame");

        // 2. Get the texture from the acquired desktop resource
        ID3D11Texture2D *AcquiredDesktopImage = nullptr;
        hr = DesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&AcquiredDesktopImage);
        DesktopResource->Release(); // Release the desktop resource immediately
        CHECK_HR(hr, "Failed to query ID3D11Texture2D from desktop resource");

        // 3. Get the description of the acquired texture
        D3D11_TEXTURE2D_DESC Desc;
        AcquiredDesktopImage->GetDesc(&Desc);

        // Set up a staging texture to copy the data from the GPU to CPU memory
        // Staging textures allow data to be copied between GPU-accessible and CPU-accessible memory.
        Desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; // Allow CPU to read
        Desc.Usage = D3D11_USAGE_STAGING;            // Staging resource
        Desc.BindFlags = 0;                          // Not bound to pipeline (e.g., as render target)
        Desc.MiscFlags = 0;                          // No miscellaneous flags (e.g., shared, cube map)

        ID3D11Texture2D *StagingTexture = nullptr;
        hr = ctx.pDevice->CreateTexture2D(&Desc, nullptr, &StagingTexture);
        AcquiredDesktopImage->Release(); // Release the acquired desktop image
        CHECK_HR(hr, "Failed to create staging texture");

        // 4. Copy the acquired texture data to the staging texture
        // This transfer happens on the GPU.
        ctx.pImmediateContext->CopyResource(StagingTexture, AcquiredDesktopImage);

        // 5. Map the staging texture to CPU memory
        // This makes the GPU data accessible to the CPU.
        D3D11_MAPPED_SUBRESOURCE MappedResource;
        hr = ctx.pImmediateContext->Map(StagingTexture, 0, D3D11_MAP_READ, 0, &MappedResource);
        CHECK_HR(hr, "Failed to map staging texture");

        // Get pixel data pointer and pitch (row stride in bytes)
        BYTE *pixels = static_cast<BYTE *>(MappedResource.pData);
        UINT pitch = MappedResource.RowPitch;

        // --- Check for blank (entirely black) screen ---
        if (IsScreenBlack(pixels, Desc.Width, Desc.Height, pitch))
        {
            LOG("Screen detected as black, skipping screenshot.");
            ctx.pImmediateContext->Unmap(StagingTexture, 0); // Unmap before releasing
            StagingTexture->Release();
            ctx.pDesktopDupl->ReleaseFrame(); // Release the frame acquired earlier
            return S_OK;                      // Not an error, just skipped
        }

        // 6. Save the pixels to a PNG file
        hr = SavePixelsToPng(ctx, outputPath, pixels, Desc.Width, Desc.Height, pitch, out_cv_image);

        // Unmap the resource before releasing
        ctx.pImmediateContext->Unmap(StagingTexture, 0);
        StagingTexture->Release(); // Release the staging texture

        // 7. Release the acquired frame
        // This tells the Desktop Duplication API that we are done with the current frame.
        ctx.pDesktopDupl->ReleaseFrame();
        CHECK_HR(hr, "Failed to release frame");

        return hr;
    }
}