/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/qnn/op/requantize.cc
 * \brief QNN requantize operator.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../transforms/infer_layout_util.h"
#include "../../transforms/pattern_util.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(SimulatedQuantizeAttrs);

/*
 * \brief Infer shape function of SimulatedQuantize op.
 * \param types The types of input args.
 * \param num_inputs The number of inputs.
 * \param attrs The op attributes.
 * \param reporter The type reporter that sets the dtype and shapes.
 * \return True if the infer shape succeeded.
 */
bool QNNSimulatedQuantizeRel(const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 6);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);

  const SimulatedQuantizeAttrs* squantize_attrs = attrs.as<SimulatedQuantizeAttrs>();
  int axis = data->shape.size() - 1;
  Optional<Integer> optional_axis = squantize_attrs->axis;
  if (optional_axis) {
    axis = optional_axis.value()->value;
  }
  CHECK_LT(axis, static_cast<int>(data->shape.size()))
      << "axis " << squantize_attrs->axis << " is out of range";
  CHECK_GE(axis, 0) << "axis " << squantize_attrs->axis << " is out of range";

  // Check and assign types for scale and zero points.
  AssignType(types[1], DataType::Float(32), data->shape[axis], reporter);  // input_scale
  AssignType(types[2], DataType::Int(32), data->shape[axis], reporter);    // input_zero_pt
  CHECK(IsScalarType(types[3], DataType::Float(32)));  // output_scale
  CHECK(IsScalarType(types[4], DataType::Int(32)));    // output_zero_point

  const Array<tvm::PrimExpr> oshape = data->shape;
  reporter->Assign(types[5], TensorType(oshape, DataType::Float(32)));
  return true;
}

// Positional relay function to create qnn requantize operator
// used by frontend FFI.
Expr MakeSimulatedQuantize(Expr data, Expr input_scale, Expr input_zero_point,
                           Expr output_scale, Expr output_zero_point,
                           DataType in_dtype, DataType out_dtype,
                           Optional<Integer> axis, String rounding) {
  LOG(INFO) << "Make QNN SimulatedQuantize";
  auto attrs = make_object<SimulatedQuantizeAttrs>();
  attrs->in_dtype = std::move(in_dtype);
  attrs->out_dtype = std::move(out_dtype);
  attrs->axis = axis;
  attrs->rounding = std::move(rounding);
  static const Op& op = Op::Get("qnn.simulated_quantize");
  return Call(op, {data, input_scale, input_zero_point, output_scale, output_zero_point},
              Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.simulated_quantize")
    .set_attrs_type<SimulatedQuantizeAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "The quantized input tensor.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("QNNSimulatedQuantize", QNNSimulatedQuantizeRel);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.simulated_quantize").set_body_typed(MakeSimulatedQuantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
