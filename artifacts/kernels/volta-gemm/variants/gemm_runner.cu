#include "gemm_baselines.cuh"
#include "gemm_contract.hpp"
#include "gemm_e09_smoke.cuh"
#include "volta_ptx_gemm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void require_cublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(status));
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    require_cuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return data_; }
  const T* get() const { return data_; }
 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

struct Options {
  std::string case_id;
  std::string input;
  std::string reference;
  std::string implementation;
  std::string mode = "correctness";
  int stages = 1;
  int warmups = 2;
  int repetitions = 11;
  bool bias_relu = false;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
      return argv[i];
    };
    if (arg == "--case") o.case_id = value("--case");
    else if (arg == "--input") o.input = value("--input");
    else if (arg == "--reference") o.reference = value("--reference");
    else if (arg == "--impl") o.implementation = value("--impl");
    else if (arg == "--mode") o.mode = value("--mode");
    else if (arg == "--stages") o.stages = std::stoi(value("--stages"));
    else if (arg == "--warmups") o.warmups = std::stoi(value("--warmups"));
    else if (arg == "--repetitions") o.repetitions = std::stoi(value("--repetitions"));
    else if (arg == "--bias-relu") o.bias_relu = true;
    else throw std::invalid_argument("unknown argument: " + arg);
  }
  if (o.case_id.empty() || o.input.empty() || o.implementation.empty())
    throw std::invalid_argument("--case, --input, and --impl are required");
  if (o.implementation != "cublas" && o.implementation != "cutlass" && o.implementation != "ptx")
    throw std::invalid_argument("--impl must be cublas, cutlass, or ptx");
  if (o.mode != "correctness" && o.mode != "time" && o.mode != "single")
    throw std::invalid_argument("--mode must be correctness, time, or single");
  if (o.mode == "correctness" && o.reference.empty())
    throw std::invalid_argument("--reference is required for correctness mode");
  if (o.stages != 1 && o.stages != 2) throw std::invalid_argument("--stages must be 1 or 2");
  if (o.warmups < 0 || o.repetitions <= 0) throw std::invalid_argument("invalid timing counts");
  if (o.bias_relu && o.implementation == "ptx" && o.mode == "time" && o.repetitions < 1)
    throw std::invalid_argument("invalid fused timing setup");
  return o;
}

kwv_gemm::GemmTile tile_for(const kwv_gemm::Shape& shape) {
  if (std::string(shape.id) == "large_dense") return kwv_gemm::GemmTile::Large64;
  if (std::string(shape.id) == "small_m") return kwv_gemm::GemmTile::SmallM16x32;
  if (std::string(shape.id) == "ragged") return kwv_gemm::GemmTile::Ragged32x64;
  throw std::invalid_argument("no custom tile configured for case");
}

int main_impl(const Options& o) {
  const auto shape = kwv_gemm::shape_by_id(o.case_id);
  const auto inputs = kwv_gemm::read_inputs(shape, o.input);
  std::unique_ptr<kwv_gemm::Oracle> reference;
  if (o.mode == "correctness")
    reference = std::make_unique<kwv_gemm::Oracle>(
        kwv_gemm::read_oracle_file(shape, o.reference));
  const std::size_t output_count = static_cast<std::size_t>(shape.m) * shape.n;
  DeviceBuffer<std::uint16_t> device_a(inputs.a_bits.size());
  DeviceBuffer<std::uint16_t> device_b(inputs.b_bits.size());
  DeviceBuffer<float> device_bias(shape.n);
  DeviceBuffer<float> device_c(output_count);
  DeviceBuffer<float> device_d(output_count);
  require_cuda(cudaMemcpy(device_a.get(), inputs.a_bits.data(), inputs.a_bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "copy represented A");
  require_cuda(cudaMemcpy(device_b.get(), inputs.b_bits.data(), inputs.b_bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "copy represented B");
  require_cuda(cudaMemcpy(device_bias.get(), inputs.bias.data(), inputs.bias.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy represented bias");

  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
  cublasHandle_t cublas = nullptr;
  require_cublas(cublasCreate(&cublas), "create cuBLAS handle");
  int cublas_version = 0;
  require_cublas(cublasGetVersion(cublas, &cublas_version), "query cuBLAS API version");
  require_cublas(cublasSetStream(cublas, stream), "bind cuBLAS stream");
  std::unique_ptr<kwv_gemm::CutlassRunner> cutlass;
  if (o.implementation == "cutlass") {
    cutlass = std::make_unique<kwv_gemm::CutlassRunner>(
        shape.m, shape.n, shape.k, device_a.get(), device_b.get(), device_c.get(), stream);
  }

  auto invoke = [&]() {
    cudaError_t cuda_status = cudaSuccess;
    if (o.implementation == "cublas") {
      require_cublas(kwv_gemm::launch_cublas_gemm(cublas, shape.m, shape.n, shape.k,
                                                  device_a.get(), device_b.get(), device_c.get()),
                     "cuBLAS GEMM");
    } else if (o.implementation == "cutlass") {
      if (cutlass->run() != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS GEMM launch failed");
    } else {
      const auto tile = tile_for(shape);
      cuda_status = kwv_gemm::launch_ptx_gemm(
          device_a.get(), device_b.get(), o.bias_relu ? device_bias.get() : nullptr,
          o.bias_relu ? device_d.get() : device_c.get(),
          shape.m, shape.n, shape.k, tile, o.stages, o.bias_relu, stream);
      require_cuda(cuda_status, "custom PTX GEMM launch");
    }
    if (o.bias_relu && o.implementation != "ptx") {
      require_cuda(kwv_gemm::launch_bias_relu(device_c.get(), device_bias.get(), device_d.get(),
                                              shape.m, shape.n, stream),
                   "separate bias/ReLU launch");
    }
  };

  const float* final_device_output = o.bias_relu && o.implementation != "ptx"
                                         ? device_d.get()
                                         : (o.bias_relu ? device_d.get() : device_c.get());
  if (o.mode == "correctness") {
    invoke();
    require_cuda(cudaStreamSynchronize(stream), "synchronize correctness run");
    std::vector<float> actual(output_count);
    require_cuda(cudaMemcpy(actual.data(), final_device_output, output_count * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy correctness output");
    const auto check = kwv_gemm::check_output(inputs, *reference, actual, o.bias_relu);
    kwv_gemm::write_check_json(std::cout, shape, check, o.bias_relu,
                               o.implementation, cublas_version,
                               o.implementation == "ptx" ? o.stages : 0);
    require_cublas(cublasDestroy(cublas), "destroy cuBLAS handle");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return check.failed_outputs == 0 ? 0 : 3;
  }

  if (o.mode == "single") {
    invoke();
    require_cuda(cudaStreamSynchronize(stream), "synchronize selected kernel");
    std::cout << "{\"format\":\"KERNELWIKI-GEMM-LAUNCH/1\",\"case_id\":\""
              << shape.id << "\",\"implementation\":\"" << o.implementation
              << "\",\"operator\":\"" << (o.bias_relu ? "gemm_bias_relu" : "gemm")
              << "\",\"cublas_api_version\":" << cublas_version
              << ",\"shared_stages\":"
              << (o.implementation == "ptx" ? o.stages : 0) << "}\n";
    require_cublas(cublasDestroy(cublas), "destroy cuBLAS handle");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
  }

  for (int i = 0; i < o.warmups; ++i) invoke();
  require_cuda(cudaStreamSynchronize(stream), "synchronize warmups");
  cudaEvent_t start = nullptr, stop = nullptr;
  require_cuda(cudaEventCreate(&start), "create start event");
  require_cuda(cudaEventCreate(&stop), "create stop event");
  std::vector<double> samples_us;
  samples_us.reserve(o.repetitions);
  for (int i = 0; i < o.repetitions; ++i) {
    require_cuda(cudaEventRecord(start, stream), "record start event");
    invoke();
    require_cuda(cudaEventRecord(stop, stream), "record stop event");
    require_cuda(cudaEventSynchronize(stop), "wait for timing sample");
    float milliseconds = 0.0f;
    require_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "read timing sample");
    samples_us.push_back(static_cast<double>(milliseconds) * 1000.0);
  }
  std::cout << std::setprecision(17) << "{\"format\":\"KERNELWIKI-GEMM-TIMING/1\",\"case_id\":\""
            << shape.id << "\",\"implementation\":\"" << o.implementation
            << "\",\"operator\":\"" << (o.bias_relu ? "gemm_bias_relu" : "gemm")
            << "\",\"cublas_api_version\":" << cublas_version
            << ",\"shared_stages\":"
            << (o.implementation == "ptx" ? o.stages : 0) << ",\"samples_us\":[";
  for (std::size_t i = 0; i < samples_us.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << samples_us[i];
  }
  std::cout << "]}\n";
  require_cuda(cudaEventDestroy(stop), "destroy stop event");
  require_cuda(cudaEventDestroy(start), "destroy start event");
  require_cublas(cublasDestroy(cublas), "destroy cuBLAS handle");
  require_cuda(cudaStreamDestroy(stream), "destroy stream");
  return 0;
}

int onehot_multitile_smoke() {
  constexpr int m = 128, n = 128, k = 4;
  std::vector<std::uint16_t> a(static_cast<std::size_t>(m) * k, 0u);
  std::vector<std::uint16_t> b(static_cast<std::size_t>(k) * n);
  for (int row = 0; row < m; ++row)
    a[static_cast<std::size_t>(row) * k + (row & 3)] =
        kwv_gemm::exact_quantized_half(1, 0);
  for (int inner = 0; inner < k; ++inner)
    for (int col = 0; col < n; ++col)
      b[static_cast<std::size_t>(inner) * n + col] =
          kwv_gemm::exact_quantized_half((inner + 1) * 128 + col, 0);

  DeviceBuffer<std::uint16_t> device_a(a.size()), device_b(b.size());
  DeviceBuffer<float> device_c(static_cast<std::size_t>(m) * n);
  require_cuda(cudaMemcpy(device_a.get(), a.data(), a.size() * sizeof(a[0]),
                          cudaMemcpyHostToDevice), "copy one-hot A");
  require_cuda(cudaMemcpy(device_b.get(), b.data(), b.size() * sizeof(b[0]),
                          cudaMemcpyHostToDevice), "copy labeled B");
  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create one-hot stream");
  for (int stages : {1, 2}) {
    require_cuda(kwv_gemm::launch_ptx_gemm(device_a.get(), device_b.get(), nullptr,
                                           device_c.get(), m, n, k,
                                           kwv_gemm::GemmTile::Large64,
                                           stages, false, stream),
                 "launch multi-tile one-hot PTX GEMM");
    require_cuda(cudaStreamSynchronize(stream), "synchronize multi-tile one-hot GEMM");
    std::vector<float> output(static_cast<std::size_t>(m) * n);
    require_cuda(cudaMemcpy(output.data(), device_c.get(), output.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy multi-tile one-hot result");
    for (int row = 0; row < m; ++row)
      for (int col = 0; col < n; ++col) {
        const float expected = static_cast<float>(((row & 3) + 1) * 128 + col);
        if (output[static_cast<std::size_t>(row) * n + col] != expected) {
          cudaStreamDestroy(stream);
          throw std::runtime_error("multi-tile one-hot GEMM differs at stage " +
                                   std::to_string(stages) + ", row " +
                                   std::to_string(row) + ", col " +
                                   std::to_string(col));
        }
      }
  }
  require_cuda(cudaStreamDestroy(stream), "destroy one-hot stream");
  std::cout << "{\"format\":\"KERNELWIKI-GEMM-ONEHOT/1\",\"m\":128,\"n\":128,\"k\":4,\"stages_checked\":[1,2],\"all_outputs_exact\":true}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--e09-smoke")
      return kwv_gemm::run_e09_mapping_smoke() ? 0 : 2;
    if (argc == 2 && std::string(argv[1]) == "--onehot-multitile-smoke")
      return onehot_multitile_smoke();
    return main_impl(parse(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "gemm_runner: " << error.what() << '\n';
    return 1;
  }
}
