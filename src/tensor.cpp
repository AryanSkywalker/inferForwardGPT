#include <iostream>
#include <vector>
#include <memory>

#include "tensor.h"

// allocates a zero-filled buffer of the given shape
Tensor::Tensor(std::vector<int> shape): shape_{shape}, owned_{std::make_unique<float[]> (numel())}, data_{owned_.get()} {} 

// non-owning view into an existing buffer
Tensor::Tensor(float* data, std::vector<int> shape): shape_{shape}, owned_{nullptr}, data_{data} {} 

// mainstream constructors
// copy constructor
Tensor::Tensor(const Tensor& other): shape_{other.shape()}, owned_{std::make_unique<float[]> (numel())}, data_{owned_.get()} {
	for (int i=0; i<numel(); ++i) at(i) = other.at(i);

}


// copy assignment
Tensor& Tensor::operator=(const Tensor& data) {
	shape_ = data.shape();
	owned_ = std::make_unique<float[]> (numel());
	data_ = owned_.get();

	for (int i=0; i<numel(); ++i) at(i) = data.at(i);

	return *this;

} 

// move constructor
Tensor::Tensor(Tensor&& other) noexcept: shape_{other.shape_}, owned_{std::move(other.owned_)}, data_{other.data_} {
	other.data_ = nullptr;

}

// move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
	if (this == &other) return *this;

	shape_ = other.shape();
	owned_ = std::move(other.owned_);
	data_ = other.data_;

	other.data_ = nullptr;

	return *this;

}


// member functions
Tensor Tensor::view(std::vector<int> new_shape) const {
	Tensor new_tensor {data_, new_shape};
	if (new_tensor.numel() != numel()) std::cout << "there has been a disturbance in the force...\n\n";

	return new_tensor;

}


// helper members
float* Tensor::data() {return data_;}

const float* Tensor::data() const {return data_;}

int Tensor::numel () const {
	int res {1};
	for (int val : shape_) res *= val;

	return res;

}

const std::vector<int>& Tensor::shape() const {return shape_;}

// at access functions
float& Tensor::at(int i) { return data_[i]; }
float& Tensor::at(int i, int j) { return data_[i * shape_[1] + j]; }
float& Tensor::at(int i, int j, int k) { return data_[(i * shape_[1] + j) * shape_[2] + k]; }

const float& Tensor::at(int i) const { return data_[i]; }
const float& Tensor::at(int i, int j) const { return data_[i * shape_[1] + j]; }
const float& Tensor::at(int i, int j, int k) const { return data_[(i * shape_[1] + j) * shape_[2] + k]; }

bool Tensor::empty() const {
	if (!data_) return true;
	else return false;	

}
