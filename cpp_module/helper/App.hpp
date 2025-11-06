#ifndef AGENT_LIVE_APP_H
#define AGENT_LIVE_APP_H

#include "Yolo.hpp"

// --- Constants ---
constexpr int TARGET_FPS = 30;
constexpr int FRAME_DELAY_MS = 1000 / TARGET_FPS;
constexpr int MAX_CONSECUTIVE_FAILURES = 5;
constexpr auto REINITIALIZE_DELAY = std::chrono::seconds(2);

inline void appRuntime() {

}

bool initYolo(YOLO& model);
#endif