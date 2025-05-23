#include "time.h"

#include <chrono>  // 提供了处理时间和日期的功能，主要用于高精度计时和时间测量

uint64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}