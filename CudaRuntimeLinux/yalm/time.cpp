#include "time.h"

#include <chrono>  // �ṩ�˴���ʱ������ڵĹ��ܣ���Ҫ���ڸ߾��ȼ�ʱ��ʱ�����

uint64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}