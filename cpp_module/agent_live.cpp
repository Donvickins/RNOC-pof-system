#if (WIN32)
#include "dxdiag.hpp"
#include "Yolo.hpp"
#include "utils.hpp"
#include <cstdlib>

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    LOG("Starting continuous screen capture...");
    LOG("Press Ctrl+C or ESC in the window to stop.");

    if (!setUpEnv())
        return -1;

    const int targetFps = 30;
    const int frameDelayMs = 1000 / targetFps;
    long long frameCount = 0;
    bool quit = false;

    cv::dnn::Net yolo_net;
    std::vector<std::string> class_names_vec;

    const std::string YOLO_MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolo11l.onnx").generic_string();
    const std::string CLASS_NAMES_PATH = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();

    // Create a resizable OpenCV window before the main loop
    std::string windowName = "Live Feed DXGI";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(windowName, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName, 1280, 720);

    cv::ocl::setUseOpenCL(true);

    HARDWARE_INFO hw_info;

    checkGPU(hw_info);
    hardwareSummary(hw_info);

    while (!quit)
    {
        DG::DXGIContext ctx;
        LOG("Attempting to initialize DXGI...");
        if (!DG::InitializeDXGI(ctx))
        {
            LOG_ERR("DXGI Initialization failed. Retrying in 2 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        LOG("DXGI Initialized successfully.");

        // Re-initialize YOLO network after DXGI is ready
        // Clear previous class names before loading new ones to avoid accumulation if setup is retried.
        class_names_vec.clear();
        if (!setupYoloNetwork(yolo_net, YOLO_MODEL_PATH, CLASS_NAMES_PATH, class_names_vec, hw_info))
        {
            LOG_ERR("Failed to setup YOLO network. Cleaning up DXGI and retrying.");
            CleanupDXGI(ctx); // Cleanup DXGI if YOLO setup fails
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        int width = 0, height = 0;
        std::vector<BYTE> pixelBuffer;
        bool duplication_active = true;
        int consecutive_failures = 0;
        const int MAX_CONSECUTIVE_FAILURES = 5;

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

                // Add a small delay before retrying
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            consecutive_failures = 0; // Reset failure counter on success

            if (!pixelBuffer.empty())
            {
                frameCount++;
                cv::Mat frame(height, width, CV_8UC4, pixelBuffer.data());
                cv::Mat frame_bgr;
                cv::cvtColor(frame, frame_bgr, cv::COLOR_BGRA2BGR);

                // Process frame with YOLOv11
                try
                {
                    processFrameWithYOLO(frame_bgr, yolo_net, class_names_vec);
                }
                catch (const cv::Exception &e)
                {
                    LOG_ERR("OpenCV error during YOLO processing: " << e.what());
                    LOG_ERR("Attempting to re-initialize DXGI and YOLO due to OpenCV error during processing.");
                    duplication_active = false; // Force re-initialization of DXGI and YOLO
                    break;
                }
                catch (const std::exception &e)
                {
                    LOG_ERR("Error during YOLO processing: " << e.what());
                    duplication_active = false; // Force re-initialization for other std exceptions too
                    break;
                }

                if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) >= 1)
                {
                    cv::imshow(windowName, frame_bgr);
                }

                int key = cv::waitKey(1);
                // Check for ESC key
                if (key == 27)
                {
                    duplication_active = false;
                    quit = true;
                }
                // Check if window was closed
                if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
                {
                    duplication_active = false;
                    quit = true;
                }

                // Maintain target frame rate
                auto endTime = std::chrono::high_resolution_clock::now();
                auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
                if (elapsedTime < frameDelayMs)
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
        // yolo_net = cv::dnn::Net(); // Optionally clear the network object if it's large and reloaded.
        // setupYoloNetwork will reassign it anyway.
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