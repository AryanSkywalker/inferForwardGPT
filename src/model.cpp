#include <cmath>
#include <iostream>
#include <vector>

#include "model.h"
#include "ops.h"
#include "tensor.h"

void mlp_block(const Tensor& x, const MLPWeights& w, Tensor& out) {
	Tensor fc {x.shape()};
	Tensor fc_b {x.shape()};
	Tensor act {x.shape()};
	Tensor proj {x.shape()};

	matmul(x, w.c_fc_w, fc);
	add_bias(fc, w.c_fc_b, fc_b);
	gelu(fc_b, act);
	matmul(act, w.c_proj_w, proj);
	add_bias(proj, w.c_proj_b, out);
} 

void attention_block(const Tensor& x, const AttentionWeights& w, Tensor& out) {
	// NOTE: missing semicolon
	int T = x.shape()[0];

	Tensor qkv {x.shape()};
	Tensor qkv_b {x.shape()};
	Tensor q {x.shape()}, k {x.shape()}, v {x.shape()};

	// project x [T x 768] to QKV space [T x 2304]
	matmul(x, w.c_attn_w, qkv);
	add_bias(qkv, w.c_attn_b, qkv_b);

	// divide the aggregate into q, k and v tensors
	for (int i=0; i<T; ++i) {
		for (int j=0; j<768; ++j) {
			// NOTE: data() returns float* so data()[idx] is already a float — no * needed
			// NOTE: third line was writing into q again instead of v
			q.data()[i*768 + j] = qkv_b.data()[i*2304 + j];
			k.data()[i*768 + j] = qkv_b.data()[i*2304 + 768 + j];
			v.data()[i*768 + j] = qkv_b.data()[i*2304 + 1536 + j];
		}
	}

	// view the qkv tensors, into 12 heads, each having it's own attention subspace
	Tensor q_re = q.view({T, 12, 64}), k_re = k.view({T, 12, 64}), v_re = v.view({T, 12, 64});

	// NOTE: missing std:: prefix
	std::vector<Tensor> attention_weights {};

	// for each head, we do it's own attention subspace
	for (int h=0; h<12; ++h) {
		// isolate into each head
		Tensor q_h {{T, 64}}, k_h {{T, 64}}, v_h {{T, 64}};

		for (int t=0; t<T; ++t) {
			for (int i=0; i<64; ++i) {
				// NOTE: q_re layout is [T, 12, 64] row-major, so [t, h, i] -> t*12*64 + h*64 + i
				// original index h*T*64 + 64*t was wrong and missing +i entirely
				q_h.data()[t * 64 + i] = q_re.data()[t*12*64 + h*64 + i];
				k_h.data()[t * 64 + i] = k_re.data()[t*12*64 + h*64 + i];
				v_h.data()[t * 64 + i] = v_re.data()[t*12*64 + h*64 + i];
			}
		}

		// NOTE: view({64, T}) is not a transpose — it reinterprets the flat buffer without
		// rearranging bytes, giving wrong data. transpose() actually swaps rows and cols.
		Tensor k_h_t {{64, T}};
		transpose(k_h, k_h_t);

		Tensor attended {{T, T}};
		matmul(q_h, k_h_t, attended);

		// NOTE: same float* dereference issue — no * needed
		for (int i=0; i<attended.numel(); ++i) attended.data()[i] /= std::sqrt(64.0f);

		// NOTE: causal mask should start at t+1, not t — a token is allowed to attend to itself
		for (int t=0; t<T; ++t)
			for (int i=t+1; i<T; ++i) attended.data()[t * T + i] = -1e10f;

		Tensor weights {attended.shape()};
		softmax(attended, weights);

		Tensor head_out {{T, 64}};
		// NOTE: weighted sum of V gives the head output [T x 64]
		matmul(weights, v_h, head_out);

		attention_weights.push_back(head_out);
	}

	// concatenate each head output [T x 64] into [T x 768]
	Tensor attention_heads {{T, 768}};

	for (int h=0; h<12; ++h) {
		for (int t=0; t<T; ++t) {
			for (int i=0; i<64; ++i) {
				// NOTE: head h must write to columns h*64..h*64+63, not 0..63 every time
				attention_heads.data()[768 * t + h * 64 + i] = attention_weights[h].data()[64 * t + i];
			}
		}
	}

	// output projection: [T x 768] -> [T x 768]
	// NOTE: this belongs here after concat, not inside the per-head loop —
	// c_proj_w is [768 x 768] and only makes sense on the full concatenated output
	Tensor proj {{T, 768}};
	matmul(attention_heads, w.c_proj_w, proj);
	add_bias(proj, w.c_proj_b, out);
}


void transformer_block(const Tensor& x, const BlockWeights& w, Tensor& out) {
	// layer_norm is applied pre-block, not after transformation

	Tensor normed {x.shape()};
	layer_norm(x, w.ln1_w, w.ln1_b, normed);

	// attention is applied after that

	Tensor attended_normed {x.shape()};
	attention_block(normed, w.attnWeights, attended_normed);

	// residual connection is established
	
	Tensor layer_1_fin {x.shape()};
	add(x, attended_normed, layer_1_fin);

	// layer_norm again
	Tensor normed_2 {x.shape()};
	layer_norm(layer_1_fin, w.ln2_w, w.ln2_b, normed_2);

	// MLP layer is established
	Tensor mlp_out {x.shape()};
	mlp_block(normed_2, w.mlp, mlp_out);

	// residual connection is reestablished
	add(layer_1_fin, mlp_out, out);
}

