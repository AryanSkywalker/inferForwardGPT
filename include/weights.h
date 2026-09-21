#pragma once
#include <string>
#include <vector>
#include <unordered_map>

#include "tensor.h"

struct WeightTensor {
	std::string name;
	std::vector<int> shape;
	std::vector<float> data;
	int numel() const;

};

std::unordered_map<std::string, Tensor> load_weights(const std::string& weights_dir);

