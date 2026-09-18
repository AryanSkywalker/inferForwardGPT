#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>

#include "weights.h"

int WeightTensor::numel () const {
	int res {1};
	for (int val : shape) res *= val;

	return res;

}

std::unordered_map<std::string, WeightTensor> load_weights(const std::string& weights_dir) {
	std::ifstream manifest {weights_dir + "/manifest.txt"};
	std::unordered_map<std::string, WeightTensor> res {};
	if (!manifest) std::cout << "Not found manifest file\n\n";
	
	
	for (std::string line {}; std::getline(manifest, line);) {
		
		WeightTensor tensor {};

		std::vector<std::string> tokens {};
		std::istringstream reader {line};

		for (std::string token {}; reader >> token;) tokens.push_back(token);

		std::stringstream ss {tokens[1]};
		for (std::string val {}; std::getline(ss, val, ',');) tensor.shape.push_back(std::stoi(val));
		tensor.data.resize(tensor.numel());


		// now we open the binary file and read the byte stream
		std::ifstream weight_file {weights_dir + "/" + tokens[2] + ".kin", std::ios::binary};
		if (!weight_file) std::cerr << "Weight file does not exit\n\n";

		// read raw binary bytes into the tensor's data vector
		weight_file.read(reinterpret_cast<char*>(tensor.data.data()), tensor.data.size() * sizeof(float));	

		// put into the unordered_map
		res[tokens[0]] = tensor;

	}

	return res;

}

