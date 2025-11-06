#include "Yolo.hpp"
#include <string>
#include <future>
#include <mutex>
#include <condition_variable>
#ifdef  _WIN32
#include "dxdiag.hpp"
#endif

int main()
{
#ifdef _WIN32
    DG::enableANSIColors();
#endif
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    if (!setUpEnv()) {
        std::cin.get();
        return -1;
    }

    constexpr int targetFps = 30;
    constexpr int frameDelayMs = 1000 / targetFps;
    long long frameCount = 0;
    bool quit = false;

    // Initialize webcam first with default resolution for quick start
    cv::VideoCapture webcam;
    constexpr int16_t MAX_INIT_ATTEMPTS = 3;
    bool webcam_initialized = false;

    for (int attempt = 0; attempt < MAX_INIT_ATTEMPTS && !webcam_initialized; attempt++)
    {
        if (attempt > 0)
        {
            LOG("Retrying webcam initialization...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

// cv::CAP_DSHOW
#if defined(_WIN32)
        webcam = cv::VideoCapture(0, cv::CAP_DSHOW);
#elif defined(__linux__)
        webcam = cv::VideoCapture(0 + cv::CAP_V4L2);
#endif
        if (webcam.isOpened())
        {
            // Quick check if we can get a frame
            if (cv::Mat test_frame; webcam.read(test_frame))
            {
                webcam_initialized = true;
                int actual_width = webcam.get(cv::CAP_PROP_FRAME_WIDTH);
                int actual_height = webcam.get(cv::CAP_PROP_FRAME_HEIGHT);
                LOG("Webcam initialized at default resolution: " << actual_width << "x" << actual_height);
            }
        }
    }

    if (!webcam_initialized)
    {
        errorHandler(std::format("Failed to initialize webcam after {}  attempts", MAX_INIT_ATTEMPTS));
        return -1;
    }

    LOG("Webcam Initialized successfully at default resolution");

    YOLO model = YOLO();
    model.HardwareSummary();
    try {
        model.Init();
    }
    catch (const cv::Exception& e) {
        errorHandler(std::format("Failed to load model: {}", e.msg));
        return -1;
    }catch (const std::exception& e) {
        errorHandler(std::format("Failed to Load model", e.what()));
        return -1;
    }

    // Create window after webcam is initialized
    static const std::string windowName = "Webcam Live Feed";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    // For thread-safe frame handling between YOLO processing and display
    std::mutex frame_mutex;
    std::future<void> yolo_future;
    std::condition_variable c_var;
    std::atomic<bool> frame_processed;
    cv::Mat display_frame;

    // Phase 2: Switch to high resolution after first frame
    bool high_res_initialized = false;
    int high_res_attempts = 0;

    while (!quit)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        cv::Mat frame_bgr;
        webcam >> frame_bgr;
        cv::flip(frame_bgr, frame_bgr, 1);

        if (!frame_bgr.empty())
        {
            frameCount++;
            // Try to switch to high resolution after first successful frame
            if (constexpr int MAX_HIGH_RES_ATTEMPTS = 3; !high_res_initialized && high_res_attempts < MAX_HIGH_RES_ATTEMPTS)
            {
                if (webcam.get(cv::CAP_PROP_FRAME_WIDTH) < 1280 || webcam.get(cv::CAP_PROP_FRAME_HEIGHT) < 720)
                {
                    LOG("Attempting to switch to high resolution...");
                    webcam.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                    webcam.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

                    // Verify the resolution change
                    int new_width = webcam.get(cv::CAP_PROP_FRAME_WIDTH);
                    int new_height = webcam.get(cv::CAP_PROP_FRAME_HEIGHT);

                    if (new_width >= 1280 && new_height >= 720)
                    {
                        high_res_initialized = true;
                        cv::resizeWindow(windowName, webcam.get(cv::CAP_PROP_FRAME_WIDTH), webcam.get(cv::CAP_PROP_FRAME_HEIGHT));
                        LOG("Successfully switched to high resolution: " << new_width << "x" << new_height);
                    }
                    else
                    {
                        high_res_attempts++;
                        LOG("Failed to switch to high resolution, attempt " << high_res_attempts << " of " << MAX_HIGH_RES_ATTEMPTS);
                    }
                }
                else
                {
                    high_res_initialized = true;
                }
            }

            try
            {
                frame_processed.store(false);
                yolo_future = std::async(std::launch::async, [&model, &frame_processed, &frame_mutex, &c_var, &display_frame, frame_to_process = frame_bgr.clone()]() mutable {
                    LOG("Processing frame...");
                    model.ProcessFrame(frame_to_process);
                    {
                        std::lock_guard lock(frame_mutex);
                        display_frame = frame_to_process.clone();
                    }
                    frame_processed = true;
                    frame_processed.store(true);
                    c_var.notify_one();
                });

                while (!quit) {
                    {
                        std::unique_lock lock(frame_mutex);
                        if (display_frame.empty()) {
                            c_var.wait(lock,[&frame_processed] {return frame_processed.load();});
                        };
                    }
                    handleWindow(windowName, display_frame, quit);
                    if (frame_processed)
                        break;
                }

            }
            catch (const cv::Exception &e)
            {
                LOG_ERR("OpenCV error during YOLO processing in webcam agent: " << e.what());
                quit = true; // Stop processing if YOLO fails critically
            }
            catch (const std::exception &e)
            {
                LOG_ERR("Standard error during YOLO processing in webcam agent: " << e.what());
                quit = true;
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            if (auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count(); elapsedTime < frameDelayMs)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(frameDelayMs - elapsedTime));
            }
        }
        else
        {
            LOG_ERR("Webcam Disconnected or Failed to get frames");
            quit = true;
        }
    }

    // Before exiting, wait for the final frame processing to complete.
    if (yolo_future.valid())
    {
        yolo_future.get();
    }

    webcam.release();
    cv::destroyAllWindows();
    LOG("Webcam Feed Ended");
    return 0;
}
