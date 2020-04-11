#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/zmath.cuh>

namespace at { namespace native {

void bitwise_not_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a) {
      return !a;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ~a;
      });
    });
  }
}

void expm1_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "expm1_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::expm1(a);
    });
  });
}

// We manually overload rsqrt because std::rsqrt does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t rsqrt_wrapper(scalar_t v) {
  return ::rsqrt(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> rsqrt_wrapper(thrust::complex<T> v) {
  const thrust::complex<T> one = thrust::complex<T>(1.0, 0);
  return one/thrust::sqrt(v);
}

void rsqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "rsqrt_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      // In CUDA, ::rsqrt is overloaded for float and at::Half here is implicitly cast to float.
      return rsqrt_wrapper(a);
    });
  });
}

// We manually overload sqrt because std::sqrt does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t sqrt_wrapper(scalar_t v) {
  return ::sqrt(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> sqrt_wrapper(thrust::complex<T> v) {
  return thrust::sqrt(v);
}

void sqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sqrt_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return sqrt_wrapper(a);
    });
  });
}

void sigmoid_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "sigmoid_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "sigmoid_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        scalar_t one = scalar_t(1);
        return  one / (one + std::exp(- a));
      });
    });
  });
}

void erfinv_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "erfinv_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erfinv(a);
    });
  });
}

template<typename scalar_t>
__host__ __device__ static inline scalar_t zabs_cuda_wrapper(scalar_t v) {
  return v;
}

template<typename T>
__host__ __device__ static inline T zabs_cuda_wrapper(thrust::complex<T> v) {
  return thrust::abs(v);
}

void clamp_kernel_cuda(TensorIterator& iter, Scalar min_scalar, Scalar max_scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, iter.dtype(), "clamp_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    auto min = thrust_t(min_scalar.to<scalar_t>());
    auto max = thrust_t(max_scalar.to<scalar_t>());
    gpu_kernel(iter,
      [=]GPU_LAMBDA(thrust_t a) -> thrust_t {
        return ((zabs_cuda_wrapper(a) < zabs_cuda_wrapper(min)) ?
                min : ((zabs_cuda_wrapper(a) > zabs_cuda_wrapper(max)) ? max : a));
    });
  });
}

void clamp_max_kernel_cuda(TensorIterator& iter, Scalar max_scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, iter.dtype(), "clamp_max_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    auto max = thrust_t(max_scalar.to<scalar_t>());
    gpu_kernel(iter,
      [=]GPU_LAMBDA(thrust_t a) -> thrust_t {
       return (zabs_cuda_wrapper(a) > zabs_cuda_wrapper(max)) ? max : a;
    });
  });
}

void clamp_min_kernel_cuda(TensorIterator& iter, Scalar min_scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, iter.dtype(), "clamp_min_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    auto min = thrust_t(min_scalar.to<scalar_t>());
    gpu_kernel(iter,
      [=]GPU_LAMBDA(thrust_t a) -> thrust_t {
       return (zabs_cuda_wrapper(a) < zabs_cuda_wrapper(min)) ? min : a;
    });
  });
}

void clamp_with_tensors_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, iter.dtype(), "clamp_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter,
      [=]GPU_LAMBDA(thrust_t a, thrust_t min, thrust_t max) -> thrust_t {
       return ((zabs_cuda_wrapper(a) < zabs_cuda_wrapper(min)) ?
                min : ((zabs_cuda_wrapper(a) > zabs_cuda_wrapper(max)) ? max : a));
    });
  });
}

void clamp_max_with_tensor_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, iter.dtype(), "clamp_max_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter,
      [=]GPU_LAMBDA(thrust_t a, thrust_t max) -> thrust_t {
       return (zabs_cuda_wrapper(a) > zabs_cuda_wrapper(max)) ? max : a;
    });
  });
}

void clamp_min_with_tensor_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, iter.dtype(), "clamp_min_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter,
      [=]GPU_LAMBDA(thrust_t a, thrust_t min) -> thrust_t {
       return (zabs_cuda_wrapper(a) < zabs_cuda_wrapper(min)) ? min : a;
    });
  });
}

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(expm1_stub, &expm1_kernel_cuda);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel_cuda);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel_cuda);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_cuda);
REGISTER_DISPATCH(clamp_max_stub, &clamp_max_kernel_cuda);
REGISTER_DISPATCH(clamp_min_stub, &clamp_min_kernel_cuda);
REGISTER_DISPATCH(clamp_with_tensors_stub, &clamp_with_tensors_kernel_cuda);
REGISTER_DISPATCH(clamp_max_with_tensor_stub, &clamp_max_with_tensor_kernel_cuda);
REGISTER_DISPATCH(clamp_min_with_tensor_stub, &clamp_min_with_tensor_kernel_cuda);
}}
