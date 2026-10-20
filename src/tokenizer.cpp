#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <climits>
#include <regex>

#include "tokenizer.h"

// loads the encoder.txt and merges.txt to the program and populates the encoder, decoder, bpe_ranks and bytes_to_unicode
void Tokenizer::load(std::string& dir) {
	// streams for the encoder and merges data file
	// path is defined from the hoem directory of build so encoder is at weights/encoder.txt
	std::ifstream encoder_file {dir + "/weights/encoder.txt"};
	std::ifstream merges_file {dir + "/weights/merges.txt"};

	// a line in encoder.txt is {id} {token-hex}
	// so tokenizer's encoder should be populated with this
	if (!encoder_file) {std::cout << "No encoder.txt found at " + dir + "/weights/encoder.txt";}
	if (!merges_file) {std::cout << "No merges.txt found at " + dir + "/weights/merges.txt";}

	for (std::string line {}; std::getline(encoder_file, line);) {
		std::istringstream reader {line};
		std::vector<std::string> wrds {};

		for (std::string wrd; reader >> wrd; ) wrds.push_back(wrd);

		encoder[wrds[1]] = std::stoi(wrds[0]);
		decoder[std::stoi(wrds[0])] = wrds[1];

	}

	// in merges.txt, a line is {left-hex} {right-hex} and index is rank
	int i = 0;
	for (std::string line{}; std::getline(merges_file, line); ++i) {
		bpe_ranks[line] = i;

	}

	build_byte_to_unicode(*this);
	
}

std::string unicode_from_utf8(int ch) {
	std::string s;

	s += (char)(0b11000000 | (ch >> 6));
	s += (char)(0b10000000 | (ch & 0b00111111));

	return s;

}


void build_byte_to_unicode(Tokenizer& tokenizer) {
	// GPT2 treats printable ASCII as identities, so it's their own unicode
	// 33-126 normal ASCII
	auto& mapper = tokenizer.byte_to_unicode;

	for (int i=33; i<127; ++i) mapper[i] = static_cast<char>(i);

	// 161 - 172
	for (int i=161; i<173; ++i) mapper[i] = unicode_from_utf8(i);

	// 174 - 255
	for (int i=174; i<256; ++i) mapper[i] = unicode_from_utf8(i);

	// now for control characters and SPACE
	int k = 256;
	for (int i=0; i<256; ++i) {
		if (mapper[i] == "") {
			mapper[i] = unicode_from_utf8(k);
			++k;

		} 

	}

}


std::string to_hex(const std::string& s) {
	std::string res;

	for (unsigned char c : s) {
		char buf[3];
		std::snprintf(buf, sizeof(buf), "%02x", c);
		res += buf;			

	}

	return res;

}

std::vector<std::string> bpe(const std::vector<std::string>& chars,
                              const std::unordered_map<std::string, int>& bpe_ranks) {
	if (chars.size() <= 1) return chars;

	std::vector<std::string> seq = chars;

	while (true) {
		// find the pair with the lowest rank across the current sequence
		int best_rank = INT_MAX;
		int best_pos  = -1;

		for (size_t i = 0; i < seq.size() - 1; ++i) {
			std::string key = to_hex(seq[i]) + " " + to_hex(seq[i + 1]);
			auto it = bpe_ranks.find(key);
			// rank is the map value, not the iterator position
			if (it != bpe_ranks.end() && it->second < best_rank) {
				best_rank = it->second;
				best_pos  = (int)i;
			}
		}

		// best_pos == -1 means no pair in the sequence exists in bpe_ranks → done
		if (best_pos == -1) break;

		// merge every occurrence of the winning pair in one pass
		std::string left  = seq[best_pos];
		std::string right = seq[best_pos + 1];
		std::vector<std::string> new_seq;

		size_t i = 0;
		while (i < seq.size()) {
			// merge both elements into one and skip ahead by 2
			if (i < seq.size() - 1 && seq[i] == left && seq[i + 1] == right) {
				new_seq.push_back(left + right);
				i += 2;
			} else {
				new_seq.push_back(seq[i]);
				++i;
			}
		}

		seq = new_seq;
	}

	return seq;
}


std::vector<int> Tokenizer::encode(const std::string& text) {
	// given text, we first split it according to the regex pattern of GPT-2 --> '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
	
	// NOTE: std::regex does not support unicode property escapes (\p{L}, \p{N}) —
	// replaced with [a-zA-Z] and [0-9] which covers ASCII text correctly
	std::regex pre_tokenization_pattern("'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s a-zA-Z0-9]+|\\s+(?!\\S)|\\s+");
	
	std::sregex_iterator pre_tokens_start = std::sregex_iterator(text.begin(), text.end(), pre_tokenization_pattern);
	std::sregex_iterator pre_tokens_end = std::sregex_iterator();

	
	// having the pre-tokens in-between the iterators start and end, now we perform byte-encoding
	// so the pre-token like ' kills' becomes 'G^dotkills' because ' ' in UTF-8 is 32 and mapped to that by GPT2

	std::vector<std::vector<std::string>> encoded_tokens {};

	for (std::sregex_iterator i {pre_tokens_start}; i != pre_tokens_end; ++i) {
		std::smatch match = *i;
		std::string match_str = match.str();

		// now we byte-encode this match_str

		std::vector<std::string> char_tokens {};
		for (unsigned char c : match_str) {
			char_tokens.push_back(byte_to_unicode[c]);

		}
	
		encoded_tokens.push_back(char_tokens);

	}

	// now we have a byte-encoded sequence of tokens, that we pass into byte-pair-encoding
	// to get a vector of merged subword sequence

	std::vector<std::string> post_bpe_seq {};
	for (const auto& chunk : encoded_tokens)
		for (const auto& tok : bpe(chunk, bpe_ranks))
			post_bpe_seq.push_back(tok);
	std::vector<int> ids {};
	
	// now, for the final step, we convert this subword sequence to ids from the table of look-up encoder
	for (auto subwrd = post_bpe_seq.begin(); subwrd != post_bpe_seq.end(); ++subwrd) {
		std::string hex_encoded_subwrd = to_hex(*subwrd);	
		auto it = encoder.find(hex_encoded_subwrd);
		
		if (it != encoder.end()) ids.push_back(it -> second);
		else std::cout << "no id for the subwrd -> " << *subwrd << "\n";

	}

	return ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
	// build reverse map: unicode string to original byte value
	std::unordered_map<std::string, unsigned char> unicode_to_byte;
	for (int b = 0; b < 256; ++b)
		unicode_to_byte[byte_to_unicode[b]] = (unsigned char)b;

	std::string result;

	for (int id : tokens) {
		auto it = decoder.find(id);
		if (it == decoder.end()) { std::cout << "no token for id " << id << "\n"; continue; }

		// decoder gives us the hex of the unicode token string to convert hex back to unicode string
		const std::string& hex = it->second;
		std::string unicode_token;
		for (size_t i = 0; i < hex.size(); i += 2) {
			unsigned char byte = (unsigned char)std::stoi(hex.substr(i, 2), nullptr, 16);
			unicode_token += (char)byte;
		}

		// each unicode character maps back to its original byte
		size_t i = 0;
		while (i < unicode_token.size()) {
			// try 2-byte utf-8 sequence first, then fall back to 1-byte
			std::string two = (i + 1 < unicode_token.size()) ? unicode_token.substr(i, 2) : "";
			std::string one = unicode_token.substr(i, 1);

			if (!two.empty() && unicode_to_byte.count(two)) {
				result += (char)unicode_to_byte[two];
				i += 2;
			} else {
				result += (char)unicode_to_byte[one];
				i += 1;
			}
		}
	}

	return result;
}
