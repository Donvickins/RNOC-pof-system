#include "Yolo.hpp"
#include "utils.hpp"
#include "Screenshot.hpp"
#include <chrono>
#include <thread>

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
    // Configure OpenCL to Use OpenCl Acceleration if available
    cv::ocl::setUseOpenCL(true);

    // Perform hardware checks to determine how the app would run
    HARDWARE_INFO hw_info;
    checkGPU(hw_info);
    hardwareSummary(hw_info);

    // Initialize ScreenShot
    std::string storagePath = "screenshots";
    Screenshot screenshot(storagePath);
    cv::Mat image;

    const std::chrono::seconds interval(5); // Capture every 5 seconds

    // Now initialize YOLO
    cv::dnn::Net yolo_net;
    std::vector<std::string> class_names_vec;

    // Yolo Model Path
    const std::string YOLO_MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolov8l.onnx").generic_string();
    const std::string CLASS_NAMES_PATH = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();

    LOG("Initializing YOLO network...");
    if (!setupYoloNetwork(yolo_net, YOLO_MODEL_PATH, CLASS_NAMES_PATH, class_names_vec, hw_info))
    {
        LOG_ERR("Failed to setup YOLO network");
        return -1;
    }

    // Main loop for capturing screenshots
    while (true)
    {
        try
        {
            screenshot.capture();
            image = screenshot.getImage();
            if(image.empty())
            {
                LOG_ERR("Image is empty, exiting...");
                break;
            }

            try
            {
                //processFrameWithYOLO(image, yolo_net, class_names_vec);
            }
            catch (const cv::Exception &e)
            {
                LOG_ERR("OpenCV error processing YOLO: " << e.what());
                continue;
            }
            catch (const std::exception &e)
            {
                LOG_ERR("Processing frame YOLO: " << e.what());
                break;
            }

            cv::imshow("Screenshot", image);
            cv::waitKey(1);
            if (cv::getWindowProperty("Screenshot", cv::WND_PROP_VISIBLE) < 1)
                break;

            std::this_thread::sleep_for(interval);
        }
        catch (const std::exception &e)
        {
            std::cerr << "An unexpected C++ exception occurred: " << e.what() << std::endl;
            break;
        }
        catch (...)
        {
            std::cerr << "An unknown exception occurred." << std::endl;
            break;
        }
    }
    return 0;
}
