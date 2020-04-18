#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qclamp_stub);
DEFINE_DISPATCH(qclamp_with_tensors_stub);
DEFINE_DISPATCH(qclamp_with_min_tensor_stub);
DEFINE_DISPATCH(qclamp_with_max_tensor_stub);

namespace {
Tensor quantized_clamp_impl(
    const Tensor& qx,
    optional<Scalar> min,
    optional<Scalar> max) {
  Tensor qy;
  if (min && max) {
    qclamp_stub(qx.device().type(), qx, *min, *max, qy);
  } else {
    TORCH_CHECK(
        false, "Both min and max should be specified for quantized clamp!");
  }
  return qy;
}
Tensor quantized_clamp_with_tensors_impl(
    const Tensor& qx,
    const Tensor& min,
    const Tensor& max) {
  Tensor qy;
  if (min.defined() && max.defined()) {
    qclamp_with_tensors_stub(qx.device().type(), qx, min, max, qy);
  } else {
    TORCH_CHECK(
        false, "Both min and max should be specifed for quantized clamp!");
  }
  return qy;
}
Tensor quantized_clamp_with_min_tensor_impl(
    const Tensor& qx,
    const Tensor& min,
    Scalar max) {
  Tensor qy;
  if (min.defined()) {
    qclamp_with_min_tensor_stub(qx.device().type(), qx, min, max, qy);
  } else {
    TORCH_CHECK(
        false, "Both min and max should be specifed for quantized clamp!");
  }
  return qy;
}
Tensor quantized_clamp_with_max_tensor_impl(
    const Tensor& qx,
    Scalar min,
    const Tensor& max) {
  Tensor qy;
  if (max.defined()) {
    qclamp_with_max_tensor_stub(qx.device().type(), qx, min, max, qy);
  } else {
    TORCH_CHECK(
        false, "Both min and max should be specifed for quantized clamp!");
  }
  return qy;
}
} // namespace

// at::native functions for the native_functions.yaml
Tensor quantized_clamp(
    const Tensor& qx,
    optional<Scalar> min,
    optional<Scalar> max) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_impl(qx, min, max);
  });
  return qy;
}
Tensor quantized_clamp_with_tensors(
    const Tensor& qx,
    const Tensor& min,
    const Tensor& max) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_with_tensors_impl(qx, min, max);
  });
  return qy;
}

Tensor quantized_clamp_with_min_tensor(
    const Tensor& qx,
    const Tensor& min,
    Scalar max) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_with_min_tensor_impl(qx, min, max);
  });
  return qy;
}

Tensor quantized_clamp_with_max_tensor(
    const Tensor& qx,
    Scalar min,
    const Tensor& max) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_with_max_tensor_impl(qx, min, max);
  });
  return qy;
}

// hardtanh is clamp with default min==-1.0f and default max==1.0f
Tensor quantized_hardtanh(
    const Tensor& qx,
    Scalar min,
    Scalar max) {
  Tensor qy;
  qy = quantized_clamp_impl(qx, min, max);
  return qy;
}

Tensor& quantized_hardtanh_out(
    Tensor& result,
    const Tensor& qx,
    Scalar min,
    Scalar max) {
  result = quantized_clamp_impl(qx, min, max);
  return result;
}

Tensor& quantized_hardtanh_(
    Tensor& self,
    Scalar min,
    Scalar max) {
  Tensor qy;
  qy = quantized_clamp_impl(self, min, max);
  // This can be optimized in a future PR if it becomes a bottleneck.
  self.copy_(qy);
  return self;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl("clamp", quantized_clamp);
  m.impl("clamp_with_tensors", quantized_clamp_with_tensors);
  m.impl("clamp_with_min_tensor_max_scalar", quantized_clamp_with_min_tensor);
  m.impl("clamp_with_min_scalar_max_tensor", quantized_clamp_with_max_tensor);
}

} // namespace native
} // namespace at
