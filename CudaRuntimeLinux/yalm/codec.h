#pragma once

#include "vendor/json.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

typedef uint16_t f16_t;

// TODO: Should this be narrowed down to what we actually support for model weight representation?
enum class DType {
	F32,
	F16,
	BF16,
	F8E5M2,
	F8E4M3,
	I32,
	I16,
	I8,
	U8,
};
std::string dtype_to_string(DType dtype);
size_t dtype_size(DType dtype);

struct Tensor {
	std::string name;
	DType dtype;
	std::array<int, 4> shape = { 0, 0, 0, 0 };  // 记录类型最多支持思维：BCWH
	void* data = nullptr;  // 实际数据存储空间。
	size_t size; // size in bytes (number of elements * element size)

	// Returns 0 if successful, other if failed
	// 从json加载tensor
	int from_json(const std::string& name, const json& j, void* bytes_ptr, size_t bytes_size);
};

// 自定义yaml格式
struct YALMData {
	void* data = nullptr;
	size_t size;

	json metadata;

	std::unordered_map<std::string, Tensor> tensors;

	// Initialize a YALMData object from a .yalm file which was created by `convert.py`.
	// Returns 0 if successful, other if failed
	// 从文件加载yalm格式模型
	int from_file(const std::string& filename);
};
