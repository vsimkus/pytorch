#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {


// These are just forwarding stubs

#define IMPLEMENT_UNARY_OP_PREQUEL(op)                           \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return legacy::cuda::_th_##op##_out(self, self);         \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return legacy::cuda::_th_##op##_out(result, self);       \
  }


IMPLEMENT_UNARY_OP_PREQUEL(atan)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(cosh)
IMPLEMENT_UNARY_OP_PREQUEL(erf)
IMPLEMENT_UNARY_OP_PREQUEL(erfc)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(tan)
IMPLEMENT_UNARY_OP_PREQUEL(tanh)

}}
