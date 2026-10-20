#include <iostream>
#include <unordered_map>
#include <string>
#include <cmath>
#include <cstdlib>
#include "weights.h"
#include "model.h"
#include "ops.h"
#include "tokenizer.h"

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
	std::unordered_map<std::string, Tensor> weights = load_weights(dir);	
	int res {};

	for (const auto& pair : weights) {
		std::cout << pair.first << " Shape: " << pair.second.shape() << " Length: " <<  pair.second.numel() << "\n";
		res += pair.second.numel();
	
	}	

	std::cout << "\n\nTotal Count of floats: " << res << "\n";

	/*Tensor a ({3,4});
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
	if (b.data() == d.data()) std::cout << "Yet they point to the same live buffer... isn't that amazing!\n\n\n";*/

	// --- transformer_block smoke test ---
	std::cout << "\n--- transformer_block smoke test ---\n";
	int T = 4, D = 768;

	Tensor x({T, D});
	for (int i = 0; i < x.numel(); ++i) x.data()[i] = ((float)rand() / RAND_MAX) * 0.02f;

	// layer norm scale=1, bias=0 — avoids div-by-zero from uniform inputs
	Tensor ln_w({D}), ln_b({D});
	for (int i = 0; i < D; ++i) { ln_w.data()[i] = 1.0f; ln_b.data()[i] = 0.0f; }

	// weight matrices left zero-initialised — output will be zero but shapes must hold
	Tensor c_attn_w({D, 3*D}), c_attn_b({3*D});
	Tensor c_attn_proj_w({D, D}), c_attn_proj_b({D});
	Tensor c_fc_w({D, 4*D}), c_fc_b({4*D});
	Tensor c_fc_proj_w({4*D, D}), c_fc_proj_b({D});

	AttentionWeights aw{c_attn_w, c_attn_b, c_attn_proj_w, c_attn_proj_b};
	MLPWeights mw{c_fc_w, c_fc_b, c_fc_proj_w, c_fc_proj_b};
	BlockWeights bw{ln_w, ln_b, ln_w, ln_b, aw, mw};

	Tensor block_out({T, D});
	transformer_block(x, bw, block_out);

	// shape check
	bool shape_ok = (block_out.shape()[0] == T && block_out.shape()[1] == D);

	// nan/inf check
	bool clean = true;
	for (int i = 0; i < block_out.numel(); ++i) {
		if (std::isnan(block_out.data()[i]) || std::isinf(block_out.data()[i])) {
			clean = false; break;
		}
	}

	std::cout << "shape check : " << (shape_ok ? "PASS" : "FAIL") << "\n";
	std::cout << "nan/inf check: " << (clean   ? "PASS" : "FAIL") << "\n";

	// --- tokenizer smoke test ---
	std::string curr_dir = ".";
	std::cout << "\n--- tokenizer smoke test ---\n";
	Tokenizer tok;
	tok.load(curr_dir);
	std::vector<int> ids = tok.encode("Hello world");
	std::string roundtrip = tok.decode(ids);
	std::cout << "input   : Hello world\n";
	std::cout << "ids     : " << ids << "\n";
	std::cout << "decoded : " << roundtrip << "\n";
	std::cout << "round-trip: " << (roundtrip == "Hello world" ? "PASS" : "FAIL") << "\n";

	return 0;

}
