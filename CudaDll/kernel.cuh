// #include "cuda_runtime.h"

extern "C" __declspec(dllexport) void addWithCuda(int* c, const int* a, const int* b, unsigned int size);
