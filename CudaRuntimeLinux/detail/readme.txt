
detail库的实现通常会关注性能，利用现代 C++ 特性和编译器优化来提高运行效率。

由于是内部实现，folly/detail 中的 API 可能会随时变化，因此不建议在应用程序中直接使用这些接口。

编译后执行举例：./CudaRuntimeLinux.exe ../../Qwen2-7b-instruct-fp16.yalm -i "什么是大模型?" -m c