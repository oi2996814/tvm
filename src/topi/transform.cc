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
 * \brief Registration of transform operators
 * \file transform.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/topi/einsum.h>
#include <tvm/topi/transform.h>
#include <tvm/topi/utils.h>

#include <iostream>

#include "tvm/ir/expr.h"

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.expand_dims",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = expand_dims(args[0].cast<te::Tensor>(), args[1].cast<int>(),
                                      args[2].cast<int>());
                  })
      .def_packed("topi.transpose",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = transpose(args[0].cast<te::Tensor>(),
                                    args[1].cast<Optional<Array<Integer>>>());
                  })
      .def_packed("topi.flip",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    // pass empty seq_lengths tensor to reverse_sequence
                    *rv =
                        reverse_sequence(args[0].cast<te::Tensor>(), Tensor(), args[1].cast<int>());
                  })
      .def_packed("topi.reverse_sequence",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = reverse_sequence(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                           args[2].cast<int>());
                  })
      .def_packed("topi.reshape",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = reshape(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>());
                  })
      .def_packed("topi.sliding_window",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = sliding_window(args[0].cast<te::Tensor>(), args[1].cast<int>(),
                                         args[2].cast<Array<Integer>>(),
                                         args[3].cast<Array<Integer>>());
                  })
      .def_packed("topi.squeeze",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = squeeze(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]));
                  })
      .def_packed("topi.concatenate",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = concatenate(args[0].cast<Array<te::Tensor>>(), args[1].cast<int>());
                  })
      .def_packed("topi.stack",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = stack(args[0].cast<Array<te::Tensor>>(), args[1].cast<int>());
                  })
      .def_packed("topi.shape",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = shape(args[0].cast<te::Tensor>(), args[1].cast<DataType>());
                  })
      .def_packed("topi.ndarray_size",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = ndarray_size(args[0].cast<te::Tensor>(), args[1].cast<DataType>());
                  })
      .def_packed("topi.split",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    if (args[1].try_cast<int>()) {
                      *rv = split_n_sections(args[0].cast<te::Tensor>(), args[1].cast<int>(),
                                             args[2].cast<int>());
                    } else {
                      *rv =
                          split_indices_array(args[0].cast<te::Tensor>(),
                                              args[1].cast<Array<Integer>>(), args[2].cast<int>());
                    }
                  })
      .def_packed("topi.layout_transform",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv =
                        layout_transform(args[0].cast<te::Tensor>(), args[1].cast<std::string>(),
                                         args[2].cast<std::string>(), args[3].cast<std::string>());
                  })
      .def_packed(
          "topi.take",
          [](ffi::PackedArgs args, ffi::Any* rv) {
            if (args.size() == 4) {
              auto mode = args[3].cast<std::string>();
              int batch_dims = args[2].cast<int>();
              *rv = take(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(), batch_dims, mode);
            } else {
              ICHECK_EQ(args.size(), 5) << "topi.take expects 4 or 5 arguments";
              int batch_dims = args[2].cast<int>();
              int axis = args[3].cast<int>();
              auto mode = args[4].cast<std::string>();
              *rv =
                  take(args[0].cast<te::Tensor>(),
                       args[1].cast<ffi::Variant<te::Tensor, PrimExpr>>(), batch_dims, axis, mode);
            }
          })
      .def_packed("topi.sequence_mask",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    double pad_val = args[2].cast<double>();
                    int axis = args[3].cast<int>();
                    *rv = sequence_mask(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                        pad_val, axis);
                  })
      .def_packed("topi.where",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = where(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                args[2].cast<te::Tensor>());
                  })
      .def_packed("topi.arange",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = arange(args[0].cast<PrimExpr>(), args[1].cast<PrimExpr>(),
                                 args[2].cast<PrimExpr>(), args[3].cast<DataType>());
                  })
      .def_packed("topi.meshgrid",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = meshgrid(args[0].cast<Array<te::Tensor>>(), args[1].cast<std::string>());
                  })
      .def_packed("topi.repeat",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = repeat(args[0].cast<te::Tensor>(), args[1].cast<int>(),
                                 args[2].cast<int>());
                  })
      .def_packed("topi.tile",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = tile(args[0].cast<te::Tensor>(), args[1].cast<Array<Integer>>());
                  })
      .def_packed("topi.gather",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = gather(args[0].cast<te::Tensor>(), args[1].cast<int>(),
                                 args[2].cast<te::Tensor>());
                  })
      .def_packed("topi.gather_nd",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    int batch_dims = args[2].cast<int>();
                    *rv = gather_nd(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                    batch_dims);
                  })
      .def_packed("topi.unravel_index",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = unravel_index(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>());
                  })
      .def_packed("topi.sparse_to_dense",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv =
                        sparse_to_dense(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                                        args[2].cast<te::Tensor>(), args[3].cast<PrimExpr>());
                  })
      .def_packed("topi.matmul",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    switch (args.size()) {
                      case 2:
                        *rv = matmul(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>());
                        break;
                      case 3:
                        *rv = matmul(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                     args[2].cast<bool>());
                        break;
                      case 4:
                        *rv = matmul(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                     args[2].cast<bool>(), args[3].cast<bool>());
                        break;
                      default:
                        ICHECK(0) << "topi.matmul expects 2, 3 or 4 arguments";
                    }
                  })
      .def_packed("topi.tensordot",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    if (args.size() == 2) {
                      *rv = tensordot(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>());
                    } else if (args.size() == 3) {
                      *rv = tensordot(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                      args[2].cast<int>());
                    } else {
                      Array<PrimExpr> axes = args[3].cast<Array<PrimExpr>>();
                      *rv = tensordot(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                      args[2].cast<Array<PrimExpr>>(), axes);
                    }
                  })
      .def_packed(
          "topi.strided_slice",
          [](ffi::PackedArgs args, ffi::Any* rv) {
            Tensor x = args[0].cast<te::Tensor>();
            Array<PrimExpr> begin = args[1].cast<Array<PrimExpr>>();
            Array<PrimExpr> end = args[2].cast<Array<PrimExpr>>();
            Array<PrimExpr> strides = args[3].cast<Array<PrimExpr>>();
            Array<Integer> axes = args[4].cast<Array<Integer>>();
            bool assume_inbound = args[6].cast<bool>();
            if (IsConstIntArray(begin) && IsConstIntArray(end) && IsConstIntArray(strides) &&
                IsConstIntArray(x->shape)) {
              Array<Integer> begin_static = args[1].cast<Array<Integer>>();
              Array<Integer> end_static = args[2].cast<Array<Integer>>();
              Array<Integer> strides_static = args[3].cast<Array<Integer>>();
              auto slice_mode = args[5].cast<std::string>();
              if (axes.size()) {
                *rv = strided_slice_with_axes(x, begin_static, end_static, strides_static, axes,
                                              slice_mode);
              } else {
                *rv = strided_slice(x, begin_static, end_static, strides_static, slice_mode);
              }
            } else {
              if (axes.size()) {
                *rv = dynamic_strided_slice_with_axes(x, begin, end, strides, axes, assume_inbound);
              } else {
                *rv = dynamic_strided_slice(x, begin, end, strides, assume_inbound);
              }
            }
          })
      .def_packed("topi.dynamic_strided_slice",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    te::Tensor begin = args[1].cast<te::Tensor>();
                    te::Tensor end = args[2].cast<te::Tensor>();
                    te::Tensor strides = args[3].cast<te::Tensor>();
                    *rv = dynamic_strided_slice(args[0].cast<te::Tensor>(), begin, end, strides);
                  })
      .def("topi.relax_dynamic_strided_slice",
           [](te::Tensor x, te::Tensor begin, te::Tensor end, te::Tensor strides,
              Array<PrimExpr> output_shape) {
             return relax::dynamic_strided_slice(x, begin, end, strides, output_shape);
           })
      .def_packed("topi.one_hot",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    int depth = args[3].cast<int>();
                    int axis = args[4].cast<int>();
                    DataType dtype = args[5].cast<DataType>();
                    *rv = one_hot(args[0].cast<te::Tensor>(), args[1].cast<PrimExpr>(),
                                  args[2].cast<PrimExpr>(), depth, axis, dtype);
                  })
      .def_packed("topi.matrix_set_diag",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    int k1 = args[2].cast<int>();
                    int k2 = args[3].cast<int>();
                    bool super_diag_right_align = args[4].cast<bool>();
                    bool sub_diag_right_align = args[5].cast<bool>();
                    *rv = matrix_set_diag(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                          k1, k2, super_diag_right_align, sub_diag_right_align);
                  })
      .def("topi.adv_index",
           [](te::Tensor x, Array<te::Tensor> indices) { return adv_index(x, indices); });
});

}  // namespace topi
}  // namespace tvm
