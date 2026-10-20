#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <array>

struct Tokenizer {
	std::unordered_map<std::string, int> encoder;
	std::unordered_map<int, std::string> decoder;
	// NOTE: value is int rank, not string — rank is needed for comparison in the BPE loop
	std::unordered_map<std::string, int> bpe_ranks;
	std::array<std::string, 256> byte_to_unicode;

	void load(std::string& dir);
	std::vector<int> encode(const std::string& text);
	std::string decode(const std::vector<int>& tokens);
};

void build_byte_to_unicode(Tokenizer& tokenizer);

// NOTE: takes bpe_ranks as a parameter since it's not a member function
std::vector<std::string> bpe(const std::vector<std::string>& chars,
                              const std::unordered_map<std::string, int>& bpe_ranks);

