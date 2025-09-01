#include "Yolo.hpp"
#include <stdexcept>

YOLO::YOLO()
{
    
}

void YOLO::LoadClassNames()
{
    std::ifstream ifs(this->class_names_path);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Failed to open file");
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        this->class_names.push_back(line);
    }
    ifs.close();
}

void YOLO::SetupYoloNetwork()
{

    this->model = cv::dnn::readNetFromONNX(this->MODEL_PATH);
    if (this->model.empty())
    {
        throw std::runtime_error("Failed to load model");
    }

    this->model.enableFusion(true);

    if (this->hw_info.has_cuda)
    {
        this->model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (this->hw_info.has_amd && this->hw_info.has_opencl)
    {
        cv::ocl::setUseOpenCL(true);
        this->model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->model.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }
    else
    {
        this->model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    this->LoadClassNames();
}

void YOLO::CheckGPU()
{
    // Check CUDA (NVIDIA)
    try
    {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0)
        {
            this->hw_info.has_cuda = true;
            this->hw_info.has_nvidia = true;
        }
    }
    catch (const cv::Exception &e)
    {
        throw std::runtime_error("CUDA check failed: ");
    }

    // Check OpenCL (AMD, Intel, NVIDIA)
    if (cv::ocl::haveOpenCL())
    {
        cv::ocl::Context context;
        if (context.create(cv::ocl::Device::TYPE_ALL))
        {
            this->hw_info.has_opencl = true;
            cv::ocl::Device device = context.device(0);
            this->hw_info.gpu_name = device.name();
            this->hw_info.gpu_vendor = device.vendorName();

            // Detect vendor
            if (this->hw_info.gpu_vendor.find("AMD") != std::string::npos)
            {
                this->hw_info.has_amd = true;
            }
            else if (this->hw_info.gpu_vendor.find("Intel") != std::string::npos)
            {
                this->hw_info.has_intel = true;
            }
            else if (this->hw_info.gpu_vendor.find("NVIDIA") != std::string::npos)
            {
                this->hw_info.has_nvidia = true;
            }
        }
    }
}

void YOLO::HardwareSummary()
{
    LOG("Hardware Detection Summary");
    if (!this->hw_info.has_cuda && !this->hw_info.has_opencl)
    {
        LOG("No GPU acceleration detected. Using CPU backend.");
        return;
    }

    LOG("GPU Vendor: " << (this->hw_info.gpu_vendor.empty() ? "N/A" : this->hw_info.gpu_vendor));
    LOG("GPU Name:   " << (this->hw_info.gpu_name.empty() ? "N/A" : this->hw_info.gpu_name));

    if (this->hw_info.has_cuda)
    {
        LOG("Backend: CUDA enabled. (Optimal performance)");
    }
    else if (this->hw_info.has_opencl)
    {
        LOG("Backend: OpenCL enabled.");
        if (this->hw_info.has_nvidia)
        {
            LOG("Note: For best performance on NVIDIA GPUs, please install the CUDA Toolkit.");
        }
    }
}

void YOLO::Init()
{
    this->CheckGPU();
    try
    {
        this->SetupYoloNetwork();
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(e.what());
    }
}

void YOLO::ProcessFrame(cv::Mat &frame)
{
    if (frame.empty() || this->model.empty())
    throw std::runtime_error("Model or Frame is invalid");

    cv::Mat blob;
    try
    {
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(this->YOLO_INPUT_WIDTH, this->YOLO_INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
        this->model.setInput(blob);
    }
    catch (const cv::Exception &e)
    {
        throw std::runtime_error(e.what());
    }

    std::vector<cv::Mat> outs;
    try
    {
        this->model.forward(outs, this->model.getUnconnectedOutLayersNames());
    }
    catch (const cv::Exception &e)
    {
        throw std::runtime_error(e.what());
    }

    // --- REVISED & IMPROVED POST-PROCESSING ---

    // The output 'outs[0]' is a Mat with 3 dimensions: [batch_size, num_channels, num_proposals]
    // For a YOLOv8-style model, this is [1, 84, 8400] where 84 = 4 (box) + 80 (classes)
    if (outs.empty() || outs[0].dims != 3)
        throw std::runtime_error("Empty detection: check if model is loaded");

    cv::Mat detections = outs[0];

    // Reshape the [1, 84, 8400] output to a 2D matrix of [84, 8400]
    cv::Mat detection_matrix_transposed(detections.size[1], detections.size[2], CV_32F, detections.ptr<float>());

    // Transpose the matrix to have proposals as rows for easier iteration: [8400, 84]
    cv::Mat detection_matrix = detection_matrix_transposed.t();

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    const float x_factor = frame.cols / (float)this->YOLO_INPUT_WIDTH;
    const float y_factor = frame.rows / (float)this->YOLO_INPUT_HEIGHT;

    // Iterate over each row (each detection proposal)
    for (int i = 0; i < detection_matrix.rows; ++i)
    {
        // Get a pointer to the current row's data [cx, cy, w, h, class_score_1, class_score_2, ...]
        const float *proposal = detection_matrix.ptr<float>(i);

        // The class scores start after the 4 box coordinates
        cv::Mat scores(1, class_names.size(), CV_32F, (void *)(proposal + 4));

        cv::Point class_id_point;
        double max_score;
        cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &class_id_point);

        if (max_score > this->CONFIDENCE_THRESHOLD)
        {
            confidences.push_back((float)max_score);
            class_ids.push_back(class_id_point.x);

            // Extract box coordinates
            const float cx = proposal[0];
            const float cy = proposal[1];
            const float w = proposal[2];
            const float h = proposal[3];

            // Scale box coordinates back to the original frame size
            int left = static_cast<int>((cx - w / 2) * x_factor);
            int top = static_cast<int>((cy - h / 2) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->CONFIDENCE_THRESHOLD, this->NMS_THRESHOLD, nms_indices);

    for (int idx : nms_indices)
    {
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = (class_id < this->class_names.size()) ? this->class_names[class_id] : "Unknown";
        label += cv::format(": %.2f", confidences[idx]);
        cv::putText(frame, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
}