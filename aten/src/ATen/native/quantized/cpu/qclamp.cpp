#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
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

// Keep the registry in the anonymous namespace.
namespace {
class QClamp final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, optional<Scalar> min, optional<Scalar> max) {
    return quantized_clamp(qx, min, max);
  }
};
class QClampWithTensors final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, optional<Tensor> min, optional<Tensor> max) {
    return quantized_clamp_with_tensors(qx, min.value(), max.value());
  }
};
class QClampWithTensors_min_tensor final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, const Tensor& min, Scalar max) {
    return quantized_clamp_with_min_tensor(qx, min, max);
  }
};
class QClampWithTensors_max_tensor final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, Scalar min, const Tensor& max) {
    return quantized_clamp_with_max_tensor(qx, min, max);
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::clamp(Tensor qx, Scalar? min, Scalar? max) -> Tensor qy",
    c10::RegisterOperators::options()
        .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
        .kernel<QClamp>(DispatchKey::QuantizedCPUTensorId))
    .op("quantized::clamp_with_tensors(Tensor qx, Tensor? min=None, Tensor? max=None) -> Tensor qy",
        c10::RegisterOperators::options()
          .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
          .kernel<QClampWithTensors>(DispatchKey::QuantizedCPUTensorId))
    .op("quantized::clamp_with_min_tensor_max_scalar(Tensor qx, Tensor min, Scalar max) -> Tensor qy",
        c10::RegisterOperators::options()
          .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
          .kernel<QClampWithTensors_min_tensor>(DispatchKey::QuantizedCPUTensorId))
    .op("quantized::clamp_with_min_scalar_max_tensor(Tensor qx, Scalar min, Tensor max) -> Tensor qy",
        c10::RegisterOperators::options()
          .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
          .kernel<QClampWithTensors_max_tensor>(DispatchKey::QuantizedCPUTensorId));
} // namespace

} // namespace native
} // namespace at
