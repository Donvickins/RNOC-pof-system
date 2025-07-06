#include "Yolo.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <thread>

int main()
{
    if (!setUpEnv())
        return -1;

    LOG("Starting Webcam Feed...");
    LOG("Press CTRL + C to exit");

    const int targetFps = 30;
    int frameDelayMs = 1000 / targetFps;
    long long frameCount = 0;
    bool quit = false;

    // Initialize webcam first with default resolution for quick start
    cv::VideoCapture webcam;
    const int MAX_INIT_ATTEMPTS = 3;
    bool webcam_initialized = false;
    HARDWARE_INFO hw_info;
    checkGPU(hw_info);

    LOG("Device: " << hw_info.gpu_name);
    if (hw_info.has_cuda)
    {
        LOG("Cuda Available: Yes");
    }
    else if (hw_info.has_nvidia && !hw_info.has_cuda)
    {
        LOG("No Cuda Toolkit found, Install CUDA for best Performance")
    }

    for (int attempt = 0; attempt < MAX_INIT_ATTEMPTS && !webcam_initialized; attempt++)
    {
        if (attempt > 0)
        {
            LOG("Retrying webcam initialization...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        webcam = cv::VideoCapture(0 + cv::CAP_DSHOW);
        if (webcam.isOpened())
        {
            // Quick check if we can get a frame
            cv::Mat test_frame;
            if (webcam.read(test_frame))
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

    // Create window after webcam is initialized
    static const std::string windowName = "Webcam Live Feed";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(windowName, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName, 1280, 720);

    // Now initialize YOLO
    cv::dnn::Net yolo_net;
    std::vector<std::string> class_names_vec;
    cv::ocl::setUseOpenCL(true);

    const std::string YOLO_MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolo11l.onnx").generic_string();
    const std::string CLASS_NAMES_PATH = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();

    LOG("Initializing YOLO network...");
    if (!setupYoloNetwork(yolo_net, YOLO_MODEL_PATH, CLASS_NAMES_PATH, class_names_vec, hw_info))
    {
        LOG_ERR("Failed to setup YOLO network for webcam agent.");
        webcam.release();
        return -1;
    }

    // Phase 2: Switch to high resolution after first frame
    bool high_res_initialized = false;
    int high_res_attempts = 0;
    const int MAX_HIGH_RES_ATTEMPTS = 3;

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
            if (!high_res_initialized && high_res_attempts < MAX_HIGH_RES_ATTEMPTS)
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
                processFrameWithYOLO(flippedFrame, yolo_net, class_names_vec);
            }
            catch (const cv::Exception &e)
            {
                LOG_ERR("OpenCV error during YOLO processing in webcam agent: " << e.what());
                quit = true; // Stop processing if YOLO fails critically
            }
            catch (const std::exception &e)
            {
                LOG_ERR("Standard error during YOLO processing in webcam agent: " << e.what());
                quit = true; // Stop processing
            }

            if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) >= 1)
            {
                cv::imshow(windowName, flippedFrame);
            }

            int key = cv::waitKey(1);
            // Check for ESC key
            if (key == 27)
            {
                quit = true;
            }
            // Check if window was closed
            if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
            {
                quit = true;
            }

            // Maintain target frame rate
            auto endTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            if (elapsedTime < frameDelayMs)
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