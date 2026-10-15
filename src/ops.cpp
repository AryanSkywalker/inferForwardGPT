#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "ops.h"
#include "tensor.h"

// assuming the tensor's are just 2 dimensional in shape (M, N) x (N, K) --> (M, K)
void matmul(const Tensor& A, const Tensor& B, Tensor& R) {
	// check for dimension matching
	if (A.shape()[1] != B.shape()[0]) throw std::out_of_range("matmul: mismatch between supposed common dimension of matrices");
	int M = A.shape()[0];
	int N = A.shape()[1];
	int K = B.shape()[1];

	Tensor res {std::vector<int> {M, K}};

	for (int m=0; m<M; ++m) {
		for (int k=0; k<K; ++k) {
			float sum {};
			for (int n=0; n<N; ++n) {
				sum += A.at(m, n) * B.at(n, k);
			}
			res.at(m, k) = sum;

		} 
	}
	R = res;

}

// non-linear terminating layer
// if one-dimensional, then it is straight-forward
// for two-dimensional, then first dimension is understood as batched and dim = 0
void softmax(const Tensor& A, Tensor& R) {
	Tensor res {A.shape()};	

	// it is highly unlikely we will be served one-dimensional input, so just doing for two-dimensional input
	for (int b=0; b<res.shape()[0]; ++b) {
		float max = *std::max_element(A.data() + b*res.shape()[1], A.data() + (b+1)*res.shape()[1]);
		float sum = 0.0f;

		for (int i=0; i<res.shape()[1]; ++i) {
			float val = std::exp(A.data()[b*res.shape()[1] + i] - max);
			res.data()[b*res.shape()[1] + i] = val;
			sum += val;

		}


		for (int i=0; i<res.shape()[1]; ++i) {
			res.data()[b*res.shape()[1] + i] /= sum;

		} 				
	}	
	R = res;

}

// layer normalisation function
// for each row, we normalize along the column variant
// better than batch normalization because it avoids 'covariate effect' in gradient-based optimisation of loss
void layer_norm(const Tensor& A, const Tensor& W, const Tensor& b, Tensor& R) {
	Tensor res {A.shape()};

	for (int k=0; k<res.shape()[0]; ++k) {
		float Ex = std::accumulate(A.data() + k*res.shape()[1], A.data() + (k+1)*res.shape()[1], 0.0f) / res.shape()[1];
		float mu_sum {};

		for (int i=0; i<res.shape()[1]; ++i) mu_sum += (*(A.data() + k*res.shape()[1] + i) - Ex) * (*(A.data() + k*res.shape()[1] + i) - Ex);

		float mu = mu_sum / res.shape()[1];		
		mu = std::sqrt(mu + 1e-5f);

		float inv_std = 1.0f / mu;

		for (int i=0; i<res.shape()[1]; ++i) *(res.data() + k*res.shape()[1] + i) = (*(A.data() + k*res.shape()[1] + i) - Ex) * *(W.data() + i) * inv_std + *(b.data() + i);

	}

	R = res;

}

// activation function --> GeLU()
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(const Tensor& A, Tensor& R) {
	Tensor res(A.shape());
	float tan_const = 0.7978845608f;

	for (int i = 0; i < A.numel(); ++i) {
		float x = *(A.data() + i);
		*(res.data() + i) = 0.5f * x * (1.0f + std::tanh(tan_const * (x + 0.044715f * x * x * x)));
	}

	R = std::move(res);
}


// adder for element-wise addition (same shape)
void add(const Tensor& A, const Tensor& B, Tensor& R) {
	Tensor res (A.shape());

	for (int i=0; i<A.numel(); ++i) *(res.data() + i) = *(A.data() + i) + *(B.data() + i);
	R = std::move(res);

}

// swaps rows and cols of a 2D matrix [M x N] -> [N x M]
void transpose(const Tensor& A, Tensor& R) {
	int M = A.shape()[0];
	int N = A.shape()[1];
	Tensor res({N, M});

	for (int m = 0; m < M; ++m)
		for (int n = 0; n < N; ++n)
			res.data()[n * M + m] = A.data()[m * N + n];

	R = std::move(res);
}

// broadcasts bias [D] across each row of A [T x D]
void add_bias(const Tensor& A, const Tensor& bias, Tensor& R) {
	int T = A.shape()[0];
	int D = A.shape()[1];
	Tensor res(A.shape());

	for (int t = 0; t < T; ++t)
		for (int d = 0; d < D; ++d)
			res.data()[t * D + d] = A.data()[t * D + d] + bias.data()[d];

	R = std::move(res);
}
