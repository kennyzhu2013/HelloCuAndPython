#pragma once

#include "model.h"

#include <memory>

// 要用于处理与概率分布和采样相关的功能
struct Sampler {
	int vocab_size;

	Sampler(const std::shared_ptr<Config> config, uint64_t seed);

	// Return the probability score corresponding to `logits[index]`.
	// This is equivalent to taking the softmax of the logits and returning
	// the value at index `index`.
	// 返回与 logits[index] 对应的概率分数。它通过对 logits 进行 softmax 计算来实现
	float sample_prob(int index, const InferenceState& s);
	// Return the index of the maximum value in `logits`.
	// 返回 logits 中最大值的索引
	int sample_argmax(const InferenceState& s);
	// Sample from the model's distribution with temperature.
	// 从模型的分布中进行采样，支持温度参数
	int sample(const InferenceState& s, float temperature = 1.0);
};
