#include "Yolo.hpp"
#include "Utils.hpp"
#include "Screenshot.hpp"
#include <chrono>
#include <future>

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

    constexpr std::chrono::milliseconds INTERVAL(10000);

    YOLO model = YOLO();
    model.HardwareSummary();
    LOG("Loading Model...");
    model.Init();
    LOG("Model Loaded Successfully...")
    
    // Initialize Screenshot
    std::string storagePath = "Screenshots";
    Screenshot screenshot(storagePath);
    cv::Mat image;

    bool quit{false};
    uint16_t retry_count{0};

    while (!quit)
    {
        auto start_time = std::chrono::steady_clock::now();
        try
        {

            LOG("Capturing screenshot...");
            screenshot.capture();
            image = screenshot.getImage();

            if (image.empty())
            {
                LOG_ERR("Image is empty, Retrying...");
                retry_count++;
                if (uint16_t MAX_RETRY{3}; retry_count >= MAX_RETRY)
                {
                    LOG("Image is empty after " << MAX_RETRY << " retries. Exiting...");
                    quit = true;
                    break;
                }
                continue;
            }
            retry_count = 0;

            std::future<void> process_frame = std::async(std::launch::async, [&]{model.ProcessFrame(image);});
            std::future_status status = process_frame.wait_for(std::chrono::milliseconds(10));
            do 
            {
                handleWindow("Screenshot", image, quit);
                status = process_frame.wait_for(std::chrono::milliseconds(10));
            }while(!quit && status != std::future_status::ready);
        }
        catch (const std::exception &e)
        {
            LOG_ERR("An unexpected exception occurred: " << e.what());
            quit = true;
            continue;
        }
        catch (...)
        {
            LOG_ERR("An unknown exception occurred.");
            quit = true;
            continue;
        }

        auto end_time = std::chrono::steady_clock::now();
        if (auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count(); elapsed_time < INTERVAL.count())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(INTERVAL.count() - elapsed_time));
        }
    }

    cv::destroyAllWindows();
    return 0;
}