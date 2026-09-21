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
	/*std::string dir = "weights";
	std::unordered_map<std::string, Tensor> weights = load_weights(dir);	
	int res {};

	for (const auto& pair : weights) {
		std::cout << pair.first << " Shape: " << pair.second.shape << " Length: " <<  pair.second.numel() << "\n";
		res += pair.second.numel();
	
	}	

	std::cout << "\n\nTotal Count of floats: " << res << "\n";*/

	Tensor a ({3,4});
	a.at(1,2) = 23;
	std::cout << "At pos (1, 2), the tensor holds --> " << a.at(1, 2) << "\n";
	Tensor b = std::move(a);
	if (a.empty()) std::cout << "Gawwd damn, a went poof\n";
	std::cout << "Sanity check, b at pos (1, 2) holds --> " << b.at(1, 2) << "\n";
	Tensor c = b;
	c.at(1,2) = 99;
	std::cout << "Now c pos(1, 2) holds --> " << c.at(1, 2) << "\n";
	std::cout << "For love's sake, b pos(1, 2) holds --> " << b.at(1, 2) << "\n";	

	Tensor d = b.view({2,6});
	std::cout << "New shape of b as told by d is --> " << d.shape() << "\n";
	if (b.data() == d.data()) std::cout << "Yet they point to the same live buffer... isn't that amazing!\n\n\n";

	return 0;

}
