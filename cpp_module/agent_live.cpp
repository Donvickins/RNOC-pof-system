#if (_WIN32)
#include "dxdiag.hpp"
#include "Yolo.hpp"
#include "utils.hpp"
#include <future>

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    LOG("Starting continuous screen capture...");
    LOG("Press Ctrl+C or ESC in the window to stop.");

    if (!setUpEnv())
        return -1;

    constexpr int targetFps = 30;
    constexpr int frameDelayMs = 1000 / targetFps;
    long long frameCount = 0;
    bool quit = false;

    YOLO model = YOLO();
    model.HardwareSummary();
    LOG("Loading Model...");
    model.Init();
    LOG("Model Loaded Successfully...")

    while (!quit)
    {
        DG::DXGIContext ctx;
        if (!DG::InitializeDXGI(ctx))
        {
            LOG_ERR("DXGI Initialization failed. Retrying in 2 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        LOG("DXGI Initialized successfully.");

        int width = 0, height = 0;
        std::vector<BYTE> pixelBuffer;
        bool duplication_active = true;
        int consecutive_failures = 0;
        constexpr int MAX_CONSECUTIVE_FAILURES = 5;

        while (duplication_active && !quit)
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            if (!DG::GetScreenPixelsDXGI(ctx.pDesktopDupl, ctx.pDevice, ctx.pImmediateContext, width, height, pixelBuffer))
            {
                DXGI_OUTDUPL_FRAME_INFO frameInfoCheck;
                IDXGIResource *resourceCheck = nullptr;
                HRESULT checkHr = ctx.pDesktopDupl->AcquireNextFrame(0, &frameInfoCheck, &resourceCheck);
                DG::SafeRelease(&resourceCheck);

                if (checkHr == DXGI_ERROR_ACCESS_LOST)
                {
                    LOG_ERR("Desktop Duplication access lost. Re-initializing DXGI and YOLO setup...");
                    duplication_active = false;
                    break;
                }

                consecutive_failures++;
                if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES)
                {
                    LOG_ERR("Too many consecutive GetScreenPixelsDXGI failures. Re-initializing DXGI and YOLO setup...");
                    duplication_active = false;
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            consecutive_failures = 0;

            if (!pixelBuffer.empty())
            {
                frameCount++;
                cv::Mat frame(height, width, CV_8UC4, pixelBuffer.data());
                cv::Mat frame_bgr;
                cv::cvtColor(frame, frame_bgr, cv::COLOR_BGRA2BGR);

                // Process frame with YOLO
                try
                {
                    std::future<void> process_frame = std::async(std::launch::async, [&]{model.ProcessFrame(frame_bgr);});
                    std::future_status status = process_frame.wait_for(std::chrono::milliseconds(10));
                    do 
                    {
                        handleWindow("Screenshot", frame_bgr, quit);
                        status = process_frame.wait_for(std::chrono::milliseconds(10));
                    }while(!quit && status != std::future_status::ready);
                }
                catch (const cv::Exception &e)
                {
                    LOG_ERR("OpenCV error during YOLO processing: " << e.what());
                    LOG_ERR("Attempting to re-initialize DXGI and YOLO due to OpenCV error during processing.");
                    duplication_active = false;
                    break;
                }
                catch (const std::exception &e)
                {
                    LOG_ERR("Error during YOLO processing");
                    duplication_active = false;
                    break;
                }
                
                // Maintain target frame rate
                auto endTime = std::chrono::high_resolution_clock::now();
                if (auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count(); elapsedTime < frameDelayMs)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(frameDelayMs - elapsedTime));
                }

                if (frameCount % 100 == 0)
                {
                    LOG("Processed " << frameCount << " frames via DXGI.");
                }
            }
        }
        LOG("Cleaning up DXGI context for this session.");
        DG::CleanupDXGI(ctx);
        if (!quit && !duplication_active) // If exited inner loop due to error, not user quit
        {
            LOG_ERR("DXGI session ended or failed. Attempting to re-initialize in 2 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    LOG("Screen capture stopped.");
    cv::destroyAllWindows(); // Ensure OpenCV windows are closed
    return 0;
}

#endif