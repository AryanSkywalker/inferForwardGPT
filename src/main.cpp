#include <iostream>
#include <unordered_map>
#include <string>
#include <cmath>
#include <cstdlib>
#include "weights.h"
#include "model.h"
#include "ops.h"
#include "tokenizer.h"
#include "tensor.h"

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


// takes wte and wpe of the trained GPT2 and the list of token IDS [say length T], and returns the input for the transformer network
// shape of return [T, 768]
Tensor embed(std::vector<int> ids, const Tensor& wte, const Tensor& wpe) {
	// result Tensor res
	int T = ids.size();
	Tensor res {{T, 768}};

	// for each id, we look at that id^th row in the wte Tensor
	// also the position's row is looked at in the wpe Tensor
	// both 768 length vectors are added and then put in the res
	for (int i=0; i<T; ++i) {
		int id = ids[i];
		
		for (int j=0; j<768; ++j) res.data()[i*768 + j] = wte.data()[id*768 + j] + wpe.data()[i*768 + j];

	}

	return res;
}


// full forward pass of GPT2
// this is where it all comes together
Tensor gpt2_forward(const Tensor& tokens, const std::unordered_map<std::string, Tensor>& weights) {
	Tensor passing_t {tokens};

	// first pass the tokens through 12 blocks of weights
	for (int p=0; p<12; ++p) {
		std::string prefix = "h." + std::to_string(p) + ".";

		AttentionWeights attnWt {
			weights.at(prefix + "attn.c_attn.weight"),
			weights.at(prefix + "attn.c_attn.bias"),
			weights.at(prefix + "attn.c_proj.weight"),
			weights.at(prefix + "attn.c_proj.bias")
		};

		MLPWeights mlpWt {
			weights.at(prefix + "mlp.c_fc.weight"),
			weights.at(prefix + "mlp.c_fc.bias"),
			weights.at(prefix + "mlp.c_proj.weight"),
			weights.at(prefix + "mlp.c_proj.bias")
		};

		BlockWeights blockWt {
			weights.at(prefix + "ln_1.weight"),
			weights.at(prefix + "ln_1.bias"),
			weights.at(prefix + "ln_2.weight"),
			weights.at(prefix + "ln_2.bias"),
			attnWt,
			mlpWt
		};

		// now we pass the coming in tensor through the block
		Tensor new_passing_t {passing_t.shape()};
		transformer_block(passing_t, blockWt, new_passing_t);
		passing_t = new_passing_t;

	}

	// now passing_t contains the tokens transformed via the blocks
	// passing_t is given to layer_norm
	Tensor norm_t {passing_t.shape()};
	layer_norm(passing_t, weights.at("ln_f.weight"), weights.at("ln_f.bias"), norm_t);

	// finally, we project norm_t [T x 768] into [T x 50257] space
	Tensor id_space ({1, 1});
	Tensor transposed_wte ({1, 1});
	transpose(weights.at("wte.weight"), transposed_wte);
	matmul(norm_t, transposed_wte, id_space);

	// return the logits
	return id_space;

} 


// given the logit space [T x 50257], from the last row, select the id based on argmax
int sample(const Tensor& logits) {
	int T = logits.shape()[0];
	int argmax = 0;

	for (int i=1; i<50257; ++i) if (logits.data()[(T-1)*50257 + argmax] < logits.data()[(T-1)*50257 + i]) argmax = i;

	return argmax;
}


// generates up to max_tokens new tokens given a prompt string
std::string generate(const std::string& prompt, int max_tokens,
                     const std::unordered_map<std::string, Tensor>& weights,
                     Tokenizer& tok) {
	std::vector<int> ids = tok.encode(prompt);

	for (int step = 0; step < max_tokens; ++step) {
		// embed current sequence
		Tensor x = embed(ids, weights.at("wte.weight"), weights.at("wpe.weight"));

		// full forward pass → logits [T x 50257]
		Tensor logits = gpt2_forward(x, weights);

		// pick next token greedily
		int next_id = sample(logits);

		// GPT-2 EOS token is 50256
		if (next_id == 50256) break;

		ids.push_back(next_id);
	}

	return tok.decode(ids);
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
	std::vector<int> ids = tok.encode("We live in a twilight world,");
	std::string roundtrip = tok.decode(ids);
	std::cout << "input   : We live in a twilight world, \n";
	std::cout << "ids     : " << ids << "\n";
	std::cout << "decoded : " << roundtrip << "\n";
	std::cout << "round-trip: " << (roundtrip == "We live in a twilight world," ? "PASS" : "FAIL") << "\n";


	// full pass check
	std::cout << "\n--- full pass check ---\n";
	std::string output = generate("We live in a twilight world,", 20, weights, tok);
	std::cout << output << "\n";
	

	return 0;

}
