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
 * \file cuDNN kernel calls for the forward algorithm.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>

#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

void ConvolutionForward(int mode, int format, int algo, int dims, int groups, const int pad[],
                        const int stride[], const int dilation[], const DLTensor* x,
                        const DLTensor* w, const DLTensor* y, const std::string& conv_dtype) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  SetConvDescriptors(entry_ptr, format, dims, groups, pad, stride, dilation, x->shape, w->shape,
                     y->shape, x->dtype, conv_dtype);
  // Set Device
  entry_ptr->conv_entry.device = x->device;
  // Set Algo
  entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);

  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      entry_ptr->conv_entry.fwd_algo, &workspace_size));

  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);

  // Compute convolution
  CUDNN_CALL(cudnnConvolutionForward(
      entry_ptr->handle, CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.input_desc, x->data, entry_ptr->conv_entry.filter_desc, w->data,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.fwd_algo,
      entry_ptr->conv_entry.workspace, workspace_size,
      CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.output_desc, y->data));
}

void ConvolutionBiasActivationForward(int mode, int format, int algo, int dims, int groups, int act,
                                      double coef, const int pad[], const int stride[],
                                      const int dilation[], const DLTensor* x, const DLTensor* w,
                                      const DLTensor* y, const DLTensor* bias,
                                      const std::string& conv_dtype) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  CUDNN_CALL(cudnnSetActivationDescriptor(entry_ptr->conv_entry.activation_desc,
                                          static_cast<cudnnActivationMode_t>(act),
                                          cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN, coef));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      entry_ptr->conv_entry.bias_desc, entry_ptr->conv_entry.tensor_format,
      CuDNNDataType::DLTypeToCuDNNType(bias->dtype), 1, static_cast<int>(w->shape[0]), 1, 1));

  SetConvDescriptors(entry_ptr, format, dims, groups, pad, stride, dilation, x->shape, w->shape,
                     y->shape, x->dtype, conv_dtype);
  // Set Device
  entry_ptr->conv_entry.device = x->device;
  // Set Algo
  entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);

  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      entry_ptr->conv_entry.fwd_algo, &workspace_size));

  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);

  // Compute convolution, add bias and apply activation
  CUDNN_CALL(cudnnConvolutionBiasActivationForward(
      entry_ptr->handle, CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.input_desc, x->data, entry_ptr->conv_entry.filter_desc, w->data,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.fwd_algo,
      entry_ptr->conv_entry.workspace, workspace_size,
      CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.output_desc, y->data, entry_ptr->conv_entry.bias_desc, bias->data,
      entry_ptr->conv_entry.activation_desc, entry_ptr->conv_entry.output_desc, y->data));
}

void FindAlgo(int format, int dims, int groups, const int pad[], const int stride[],
              const int dilation[], const int x_dim[], const int w_dim[], const int y_dim[],
              const std::string& data_dtype, const std::string& conv_dtype, bool verbose,
              ffi::Any* ret) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  const int full_dims = dims + 2;
  std::vector<int64_t> x_dim_int64(full_dims);
  std::vector<int64_t> w_dim_int64(full_dims);
  std::vector<int64_t> y_dim_int64(full_dims);
  for (int i = 0; i < full_dims; ++i) {
    x_dim_int64[i] = x_dim[i];
    w_dim_int64[i] = w_dim[i];
    y_dim_int64[i] = y_dim[i];
  }
  SetConvDescriptors(entry_ptr, format, dims, groups, pad, stride, dilation, x_dim_int64.data(),
                     w_dim_int64.data(), y_dim_int64.data(), StringToDLDataType(data_dtype),
                     conv_dtype);

  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_count, perf_results));

  const std::vector<std::string> fwd_algo_names{"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"};

  auto best_algo = perf_results[0].algo;
  if (verbose) {
    LOG(INFO) << "\tCUDNN Found " << returned_algo_count << " fwd algorithms, choosing "
              << fwd_algo_names[best_algo];
    for (int i = 0; i < returned_algo_count; ++i) {
      LOG(INFO) << "\t\t" << i << ") " << fwd_algo_names[perf_results[i].algo]
                << " - time: " << perf_results[i].time << " ms"
                << ", Memory: " << perf_results[i].memory;
    }
  }

  ret[0] = static_cast<int>(best_algo);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvm.contrib.cudnn.conv2d.forward",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    int mode = args[0].cast<int>();
                    int format = args[1].cast<int>();
                    int algo = args[2].cast<int>();
                    int pad_v[2], stride_v[2], dilation_v[2];
                    for (int i = 0; i < 2; i++) {
                      pad_v[i] = args[3 + i].cast<int>();
                      stride_v[i] = args[5 + i].cast<int>();
                      dilation_v[i] = args[7 + i].cast<int>();
                    }
                    auto x = args[9].cast<DLTensor*>();
                    auto w = args[10].cast<DLTensor*>();
                    auto y = args[11].cast<DLTensor*>();
                    auto conv_dtype = args[12].cast<std::string>();
                    int groups = args[13].cast<int>();

                    ConvolutionForward(mode, format, algo, 2, groups, pad_v, stride_v, dilation_v,
                                       x, w, y, conv_dtype);
                  })
      .def_packed("tvm.contrib.cudnn.conv2d+bias+act.forward",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    int mode = args[0].cast<int>();
                    int format = args[1].cast<int>();
                    int algo = args[2].cast<int>();
                    int pad_v[2], stride_v[2], dilation_v[2];
                    for (int i = 0; i < 2; i++) {
                      pad_v[i] = args[3 + i].cast<int>();
                      stride_v[i] = args[5 + i].cast<int>();
                      dilation_v[i] = args[7 + i].cast<int>();
                    }
                    int act = args[9].cast<int>();
                    double coef = args[10].cast<double>();
                    auto x = args[11].cast<DLTensor*>();
                    auto w = args[12].cast<DLTensor*>();
                    auto bias = args[13].cast<DLTensor*>();
                    auto y = args[14].cast<DLTensor*>();
                    auto conv_dtype = args[15].cast<std::string>();
                    int groups = args[16].cast<int>();

                    ConvolutionBiasActivationForward(mode, format, algo, 2, groups, act, coef,
                                                     pad_v, stride_v, dilation_v, x, w, y, bias,
                                                     conv_dtype);
                  })
      .def_packed("tvm.contrib.cudnn.conv3d.forward",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    int mode = args[0].cast<int>();
                    int format = args[1].cast<int>();
                    int algo = args[2].cast<int>();
                    int pad_v[3], stride_v[3], dilation_v[3];
                    for (int i = 0; i < 3; i++) {
                      pad_v[i] = args[3 + i].cast<int>();
                      stride_v[i] = args[6 + i].cast<int>();
                      dilation_v[i] = args[9 + i].cast<int>();
                    }
                    auto x = args[12].cast<DLTensor*>();
                    auto w = args[13].cast<DLTensor*>();
                    auto y = args[14].cast<DLTensor*>();
                    auto conv_dtype = args[15].cast<std::string>();
                    int groups = args[16].cast<int>();

                    ConvolutionForward(mode, format, algo, 3, groups, pad_v, stride_v, dilation_v,
                                       x, w, y, conv_dtype);
                  })
      .def_packed("tvm.contrib.cudnn.conv.forward_find_algo",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    int format = args[0].cast<int>();
                    int dims = args[1].cast<int>();
                    int* pad = static_cast<int*>(args[2].cast<void*>());
                    int* stride = static_cast<int*>(args[3].cast<void*>());
                    int* dilation = static_cast<int*>(args[4].cast<void*>());
                    int* x_dim = static_cast<int*>(args[5].cast<void*>());
                    int* w_dim = static_cast<int*>(args[6].cast<void*>());
                    int* y_dim = static_cast<int*>(args[7].cast<void*>());
                    auto data_dtype = args[8].cast<std::string>();
                    auto conv_dtype = args[9].cast<std::string>();
                    int groups = args[10].cast<int>();
                    bool verbose = args[11].cast<bool>();
                    FindAlgo(format, dims, groups, pad, stride, dilation, x_dim, w_dim, y_dim,
                             data_dtype, conv_dtype, verbose, ret);
                  });
});

}  // namespace contrib
}  // namespace tvm
