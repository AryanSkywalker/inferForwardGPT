#pragma once
#include <unordered_map>
#include <vector>
#include <array>

struct Tokenizer {
	std::unordered_map<std::string, int> encoder; // encoder works in the format '(hex-string) (id assigned)'
	std::unordered_map<int, std::string> decoder; // reverse of encoder
	std::unordered_map<std::string, std::string> bpe_ranks; // merger ranks from merges.txt in the format '(left-hex) (right-hex)'
	std::array<std::string, 256> byte_to_unicode; // pre-set look-up table for converting each byte to a unicode character 
	// each byte has 8 bits, so it can represent from 0 - 255. an array's position represents this number  

	// to load the encoder, decoder, bpe_ranks and byte_to_unicode
	void load(std::string& dir);
	
	// to encode given text into ids
	std::vector<int> encode(std::vector<string>& text);

	// to decode a vector of ids into a vector of strings
	std::vector<std::string> decode(std::vector<int>& tokens);

};

// given a Tokenizer structure, fill a tokenizer's byte_to_unicode
void build_byte_to_unicode(const Tokenizer& tokenizer);

// the core algorithm
std::vector<string> bpe(const std::string& token);

// encoding algorithm
std::vector<int> encode(const std::string& text);

// decoding algorithm
std::string decode(const std::vector<int> ids);

