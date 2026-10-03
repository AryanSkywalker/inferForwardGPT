#pragma once

#include "tensor.h"

// declarations for all the mathematical operations needed to perform a forward pass

void matmul(const Tensor& A, const Tensor& B, Tensor& R);
void softmax(const Tensor& A, Tensor& R);
void layer_norm(const Tensor& A, const Tensor& W, const Tensor& b, Tensor& R);
void gelu(const Tensor& A, Tensor& R);

