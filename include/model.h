#pragma once
#include "tensor.h"

struct MLPWeights {
	const Tensor& c_fc_w;
	const Tensor& c_fc_b;
	const Tensor& c_proj_w;
	const Tensor& c_proj_b;

};

struct AttentionWeights {
	const Tensor& c_attn_w;
	const Tensor& c_attn_b;
	const Tensor& c_proj_w;
	const Tensor& c_proj_b;

};

struct BlockWeights {
	const Tensor& ln1_w; const Tensor& ln1_b;
	const Tensor& ln2_w; const Tensor& ln2_b;

	AttentionWeights attnWeights;
	MLPWeights mlp;

};


void mlp_block(const Tensor& x, const MLPWeights& w, Tensor& out);

void attention_block(const Tensor& x, const AttentionWeights& w, Tensor& out);

void transformer_block(const Tensor& x, const BlockWeights& w, Tensor& out);

