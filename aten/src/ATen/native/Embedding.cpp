#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace at { namespace native {

Tensor embedding(const Tensor & weight, const Tensor & indices,
                 int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding", indices_arg, kLong);

  // TODO: use tensor.index() after improving perf
  if (indices.dim() == 1) {
    return weight.index_select(0, indices);
  }

  auto size = indices.sizes().vec();
  for (auto d : weight.sizes().slice(1)) {
    size.push_back(d);
  }
  return weight.index_select(0, indices.reshape(-1)).view(size);
}

Tensor embedding_sparse_backward(
    const Tensor & grad_, const Tensor & indices_, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  auto indices_arg = TensorArg(indices_, "indices", 2);
  checkScalarType("embedding_backward", indices_arg, kLong);

  // TODO: implement scale_grad_by_freq
  if (scale_grad_by_freq) {
    AT_ERROR(
        "embedding_backward: scale_grad_by_freq not supported with sparse gradients");
  }

  Tensor indices = indices_;
  Tensor grad = grad_;
  if (padding_idx != -1) {
    auto c = indices != padding_idx;
    indices = indices.index(c);
    grad = grad.index(c);
  }

  int64_t num_features = grad_.size(-1);
  auto weight_size = std::array<int64_t, 2>{{ num_weights, num_features }};
  auto& dense_type = grad.type();
  auto& sparse_type = dense_type.toBackend(grad.is_cuda() ? Backend::SparseCUDA : Backend::SparseCPU);

  // check if all our grad come from padding_idx
  if (grad.numel() == 0) {
    return sparse_type._sparse_coo_tensor_unsafe(indices_.type().tensor({1, 0}),
                                                 dense_type.tensor({0, num_features}),
                                                 weight_size);
  }

  auto index = indices.reshape({1, -1});
  auto values = grad.reshape({-1, num_features});
  return sparse_type._sparse_coo_tensor_unsafe(index, values, weight_size);
}

Tensor embedding_dense_backward(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {
      
      auto indices_arg = TensorArg(indices, "indices", 2);
      checkScalarType("embedding_backward", indices_arg, kLong);

      auto indices_contig = indices.contiguous();
      int64_t numel = indices_contig.numel();

      indices_contig = indices_contig.view(-1);

      Tensor counts;
      if (scale_grad_by_freq){
        counts = at::zeros(indices.numel(),grad_.type());
        counts.index_add_(0,indices_contig,at::ones(grad_.type(),indices.numel()));
      }else{
        //counts = at::ones(grad_.type(),indices.numel());
        counts = at::zeros(indices.numel(),grad_.type());
        counts.fill_(1.0);
      }

      auto freq_weight = 1 / counts.index_select(0,indices_contig);

      if(padding_idx != -1){
        //auto padded_mask = at::zeros(indices.numel(),grad_.type());
        //padded_mask.fill_(padding_idx);
        //padded_mask = padded_mask.eq(indices_contig);
        //freq_weight.masked_fill_(padded_mask, 0.0);
        auto c = (indices == padding_idx);
        freq_weight.masked_fill_(c, 0.0);
      }

      auto grad = grad_.contiguous().view({numel, grad_.size(-1)});
      auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());

      grad_weight.index_add_(0, indices_contig, grad, freq_weight);

      return grad_weight;
}

Tensor embedding_backward(
    const Tensor & grad, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (sparse) {
    return at::native::embedding_sparse_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  } else {
    return at::native::embedding_dense_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
}

Tensor & embedding_renorm_cpu_(
    Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkDim("embedding_renorm_", self_arg, 2);
  checkScalarType("embedding_renorm_", indices_arg, kLong);

  auto indices_contig = indices.contiguous();

  auto num_indices = indices.numel();
  auto data_ptr = indices_contig.data<int64_t>();
  auto sorted_indices = std::vector<int64_t>(data_ptr, data_ptr + num_indices);
  std::sort(sorted_indices.begin(), sorted_indices.end(), std::less<int64_t>());

  #pragma omp parallel for if(num_indices > 1000)
  for (int64_t i = 0; i < num_indices; i++) {
    if (i > 0 && sorted_indices[i] == sorted_indices[i - 1]) {
      continue;
    }
    auto row = self[sorted_indices[i]];
    auto norm = row.norm(norm_type).toCDouble();
    if (norm > max_norm) {
      auto scale = max_norm / (norm + 1e-7);
      row *= scale;
    }
  }

  return self;
}

}}  // namespace at::native
