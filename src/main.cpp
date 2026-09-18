#include <iostream>
#include <unordered_map>
#include <string>
#include "weights.h"

std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
    if (vec.empty()) {
        return os; 
    }
    
    os << vec[0];
    
    for (size_t i = 1; i < vec.size(); ++i) {
        os << ", " << vec[i];
    }
    
    return os;
}


int main() {
	std::string dir = "weights";
	std::unordered_map<std::string, WeightTensor> weights = load_weights(dir);	
	int res {};

	for (const auto& pair : weights) {
		std::cout << pair.first << " Shape: " << pair.second.shape << " Length: " <<  pair.second.numel() << "\n";
		res += pair.second.numel();
	
	}	

	std::cout << "\n\nTotal Count of floats: " << res << "\n";

	return 0;

}
