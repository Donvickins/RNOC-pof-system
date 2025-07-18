#include "Yolo.hpp"

bool loadClassNames(const std::string &path, std::vector<std::string> &class_names_out)
{
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open())
    {
        std::cerr << "Error: Failed to open class names file: " << path << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        class_names_out.push_back(line);
    }
    ifs.close();
    return true;
}

// In helper/classes/Yolo.cpp

bool setupYoloNetwork(cv::dnn::Net &net, const std::string &model_path, const std::string &class_names_path, std::vector<std::string> &out_class_names_vec, HARDWARE_INFO &hw_info)
{
    // 1. Load the ONNX model
    LOG("Loading YOLO model from: " << model_path);
    try
    {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty())
        {
            LOG_ERR("Failed to load YOLO model");
            return false;
        }
    }
    catch (const cv::Exception &e)
    {
        LOG_ERR("OpenCV exception during YOLO model loading: " << e.what());
        return false;
    }

    net.enableFusion(false);

    // 2. Configure the backend based on hardware detection
    if (hw_info.has_cuda)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (hw_info.has_opencl)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    // 3. Load class names
    LOG("Loading class names from: " << class_names_path);
    if (!loadClassNames(class_names_path, out_class_names_vec) || out_class_names_vec.empty())
    {
        LOG_ERR("Could not initialize network because class names failed to load.");
        return false;
    }

    LOG("Class names loaded: " << out_class_names_vec.size() << " classes.");
    LOG("YOLO network setup complete.");
    return true;
}

void processFrameWithYOLO(cv::Mat &frame, cv::dnn::Net &net, const std::vector<std::string> &class_names_list)
{
    if (frame.empty() || net.empty())
    {
        LOG_ERR("YOLO: processFrameWithYOLO called with an empty frame or network.");
        return;
    }

    cv::Mat blob;
    try
    {
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);
    }
    catch (const cv::Exception &e)
    {
        LOG_ERR("YOLO: OpenCV Exception during blob creation or setInput: " << e.what());
        throw;
    }

    std::vector<cv::Mat> outs;
    try
    {
        net.forward(outs, net.getUnconnectedOutLayersNames());
    }
    catch (const cv::Exception &e)
    {
        LOG_ERR("YOLO: OpenCV Exception during net.forward(): " << e.what());
        throw;
    }

    // --- REVISED & IMPROVED POST-PROCESSING ---

    // The output 'outs[0]' is a Mat with 3 dimensions: [batch_size, num_channels, num_proposals]
    // For a YOLOv8-style model, this is [1, 84, 8400] where 84 = 4 (box) + 80 (classes)
    if (outs.empty() || outs[0].dims != 3)
    {
        LOG_ERR("YOLO: Invalid output from network forward pass.");
        return;
    }
    cv::Mat detections = outs[0];

    // Reshape the [1, 84, 8400] output to a 2D matrix of [84, 8400]
    cv::Mat detection_matrix_transposed(detections.size[1], detections.size[2], CV_32F, detections.ptr<float>());

    // Transpose the matrix to have proposals as rows for easier iteration: [8400, 84]
    cv::Mat detection_matrix = detection_matrix_transposed.t();

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    const float x_factor = frame.cols / (float)YOLO_INPUT_WIDTH;
    const float y_factor = frame.rows / (float)YOLO_INPUT_HEIGHT;

    // Iterate over each row (each detection proposal)
    for (int i = 0; i < detection_matrix.rows; ++i)
    {
        // Get a pointer to the current row's data [cx, cy, w, h, class_score_1, class_score_2, ...]
        const float *proposal = detection_matrix.ptr<float>(i);

        // The class scores start after the 4 box coordinates
        cv::Mat scores(1, class_names_list.size(), CV_32F, (void *)(proposal + 4));

        cv::Point class_id_point;
        double max_score;
        cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &class_id_point);

        if (max_score > CONFIDENCE_THRESHOLD)
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
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);

    for (int idx : nms_indices)
    {
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = (class_id < class_names_list.size()) ? class_names_list[class_id] : "Unknown";
        label += cv::format(": %.2f", confidences[idx]);
        cv::putText(frame, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    LOG("Processed Frame successfully");
}