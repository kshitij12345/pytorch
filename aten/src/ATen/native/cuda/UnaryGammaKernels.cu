#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/Math.h>

namespace at { namespace native {

// See note [Jiterator]
const char digamma_name[] = "digamma";
void digamma_kernel_cuda(TensorIteratorBase& iter) {
  #ifdef USE_JITERATOR
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "digamma_cuda", [&]() {
      jitted_gpu_kernel</*name=*/digamma_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, digamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "digamma_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return calc_digamma(a);
      });
    });
  #endif // USE_JITERATOR
}

// See note [Jiterator]
const char trigamma_name[] = "trigamma";
void trigamma_kernel_cuda(TensorIteratorBase& iter) {
  #ifdef USE_JITERATOR
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "trigamma_cuda", [&]() {
      jitted_gpu_kernel</*name=*/trigamma_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, trigamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "trigamma_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return calc_trigamma(a);
      });
    });
  #endif // USE_JITERATOR
}

const char polygamma_name[] = "polygamma";
void polygamma_kernel_cuda(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel_cuda(iter);
  } else if (n == 1) {
    trigamma_kernel_cuda(iter);
  } else {
#ifdef USE_JITERATOR
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        iter.common_dtype(), "polygamma_cuda", [&]() {
          jitted_gpu_kernel<
              /*name=*/polygamma_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(
              iter,
              polygamma_string,
              at::cuda::jit::BinaryFuncVariant::NoScalar,
              0,
              {{"int", "n"}}, // extra args to the kernel
              &n); // pointer to the args
        });
#else
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        iter.common_dtype(), "polygamma_cuda", [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return calc_polygamma<scalar_t, /*is_cuda=*/true>(a, n);
          });
        });
#endif // USE_JITERATOR
  }
}

const char lgamma_name[] = "lgamma_kernel";
void lgamma_kernel_cuda(TensorIteratorBase& iter) {
  #ifdef USE_JITERATOR
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "lgamma_cuda", [&]() {
      jitted_gpu_kernel</*name=*/lgamma_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, lgamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "lgamma_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::lgamma(a);
      });
    });
  #endif
}

REGISTER_DISPATCH(digamma_stub, &digamma_kernel_cuda);
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel_cuda);
REGISTER_DISPATCH(lgamma_stub, &lgamma_kernel_cuda);

}} // namespace at::native
