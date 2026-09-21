#pragma once
#include <memory>
#include <vector>

class Tensor {
	std::vector<int> shape_;
	std::unique_ptr<float[]> owned_; // when tensor owns the memory, this is not null
	float* data_; // always points to the live buffer
	
	public:
	explicit Tensor(std::vector<int> shape); // allocates a zero-filled buffer of the same shape
	Tensor(float* data, std::vector<int> shape); // non-owning view into an existing buffer

	float& at(int i);
	float& at(int i, int j);
	float& at(int i, int j, int k);
	const float& at(int i) const;
	const float& at(int i, int j) const;
	const float& at(int i, int j, int k) const;	

	// constructors
	Tensor(const Tensor& other);
	Tensor& operator=(const Tensor& data);
	Tensor(Tensor&& other) noexcept;
	Tensor& operator=(Tensor&& other) noexcept;

	// defualt destruction
	~Tensor() = default;

	// reshaping the tensor (not in-place)
	Tensor view(std::vector<int> new_shape) const;
	
	float* data();
	const float* data() const;
	int numel() const;
	const std::vector<int>& shape() const;
	bool empty() const; // true if data_ is null

};
