#include "Yolo.hpp"
#include "utils.hpp"
#include "Screenshot.hpp"
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    if (!setUpEnv())
    {
        LOG_ERR("Failed to set up environment. Exiting.");
        return -1;
    }

#if (__linux__)
    if (!supportedWindowingSystem())
        return -1;
#endif

    // Perform hardware checks
    HARDWARE_INFO hw_info;
    checkGPU(hw_info);
    hardwareSummary(hw_info);

    // Initialize Screenshot
    std::string storagePath = "screenshots";
    Screenshot screenshot(storagePath);
    cv::Mat image;
    cv::Mat image_clone;

    // Initialize Window
    std::string winname = "Screenshot";
    cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::resizeWindow(winname, 1280, 720);

    // Force window initialization with a dummy image
    cv::Mat dummy(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::imshow(winname, dummy);
    for (int i = 0; i < 5; ++i) // Multiple waitKey calls to ensure GUI initialization
    {
        cv::waitKey(20);
    }
    // Interval between screenshots
    const std::chrono::milliseconds interval(10000); // 10 seconds

    // Initialize YOLO
    cv::dnn::Net yolo_net;
    std::vector<std::string> class_names_vec;
    const std::string YOLO_MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolov8l.onnx").generic_string();
    const std::string CLASS_NAMES_PATH = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();

    LOG("Initializing YOLO network...");
    if (!setupYoloNetwork(yolo_net, YOLO_MODEL_PATH, CLASS_NAMES_PATH, class_names_vec, hw_info))
    {
        LOG_ERR("Failed to setup YOLO network");
        cv::destroyAllWindows();
        return -1;
    }

    bool quit{false};
    bool yolo_processing{false};
    uint16_t MAX_RETRY{3};
    uint16_t retry_count{0};

    // Main loop for capturing screenshots and updating GUI
    while (!quit)
    {
        auto start_time = std::chrono::steady_clock::now();
        try
        {

            LOG("Capturing screenshot...");
            screenshot.capture();
            image = screenshot.getImage();
            image_clone = image.clone();

            if (image.empty())
            {
                LOG_ERR("Image is empty, Retrying...");
                retry_count++;
                if (retry_count >= MAX_RETRY)
                {
                    LOG("Image is empty after " << MAX_RETRY << " retries. Exiting...");
                    quit = true;
                    break;
                }
                continue;
            }
            retry_count = 0;

            LOG("Processing frame with YOLO...");
            yolo_processing = true;
            processFrameWithYOLO(image, yolo_net, class_names_vec);
            yolo_processing = false;

            while (yolo_processing)
            {
                if (!image_clone.empty())
                {
                    cv::imshow(winname, image_clone);
                    cv::waitKey(10);
                }
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERR("An unexpected C++ exception occurred: " << e.what());
            quit = true;
            continue;
        }
        catch (...)
        {
            LOG_ERR("An unknown exception occurred.");
            quit = true;
            continue;
        }

        cv::imshow(winname, image);
        int key = cv::waitKey(30);

        if (key == 27)
        {
            LOG("Exitting...");
            quit = true;
            break;
        }

        double visible = cv::getWindowProperty(winname, cv::WND_PROP_VISIBLE);
        if (visible <= 0) // Ignore -1 (invalid window)
        {
            LOG("Exitting...");
            quit = true;
            break;
        }

        // Maintain screenshot interval
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (elapsed_time < interval.count())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(interval.count() - elapsed_time));
        }
    }

    cv::destroyAllWindows();
    return 0;
}