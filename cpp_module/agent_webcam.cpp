#include "Yolo.hpp"
#include <string>
#include <future>

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    if (!setUpEnv())
        return -1;

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
        LOG_ERR("Failed to initialize webcam after " << MAX_INIT_ATTEMPTS << " attempts");
        return -1;
    }

    LOG("Webcam Initialized successfully at default resolution");

    YOLO model = YOLO();
    model.HardwareSummary();
    LOG("Loading Model...");
    model.Init();
    LOG("Model Loaded Successfully...")

    // Create window after webcam is initialized
    static const std::string windowName = "Webcam Live Feed";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    // Phase 2: Switch to high resolution after first frame
    bool high_res_initialized = false;
    int high_res_attempts = 0;

    while (!quit)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        cv::Mat frame_bgr;
        cv::Mat flippedFrame;
        webcam >> frame_bgr;
        cv::flip(frame_bgr, flippedFrame, 1);

        if (!flippedFrame.empty())
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
                std::future<void> process_frame = std::async(std::launch::async, [&]{model.ProcessFrame(flippedFrame);});
                std::future_status status = process_frame.wait_for(std::chrono::milliseconds(10));
                do 
                {
                    handleWindow(windowName, flippedFrame, quit);
                    status = process_frame.wait_for(std::chrono::milliseconds(10));
                }while(!quit && status != std::future_status::ready);
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

    webcam.release();
    cv::destroyAllWindows();
    LOG("Webcam Feed Ended");
    return 0;
}