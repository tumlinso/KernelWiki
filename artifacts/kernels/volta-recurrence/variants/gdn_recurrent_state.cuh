// Scratch KWV-08 kernel proposal. Authored for eventual repository adoption.
// The surrounding harness, numerical contract, and measured cases must be
// supplied by the claimed KWV-08 implementation task.
#include <cuda_runtime.h>
#include <cstddef>

namespace kwv08 {

// Conventional materialized sequential baseline: the host enqueues one
// complete state-update kernel per token on the caller's stream. Each launch
// reloads the row from global state and writes the next state before the next
// token launch. This boundary makes the state traffic observable to the
// compiler. Its operator timing must include all T launches and host enqueue
// cost; do not report a single-step latency as complete sequence time.
template <int DV, int DK>
__global__ void recurrent_step_kernel(
    const float* __restrict__ keys,       // [T, B, d_k], time-major interleave
    const float* __restrict__ values,     // [T, B, d_v]
    const float* __restrict__ queries,    // [T, B, d_k]
    const float* __restrict__ alpha,      // [T, B]
    const float* __restrict__ beta,       // [T, B]
    const int* __restrict__ lengths,      // [B], 0 <= length <= T
    float* __restrict__ states,           // [B, d_v, d_k], read/write
    float* __restrict__ outputs,          // [T, B, d_v], live prefix written
    int t,
    int T,
    int B) {
  const int seq = blockIdx.x;
  const int row = threadIdx.x;
  if (seq >= B) return;
  const int sequence_length = lengths[seq];
  // Uniform across this CTA, including ragged and zero-length sequences.
  if (t >= sequence_length || t >= T) return;

  __shared__ float k_step[DK];
  __shared__ float q_step[DK];
  __shared__ float v_step[DV];
  __shared__ float alpha_step;
  __shared__ float beta_step;

  if (row < DK) {
    const std::size_t input_base =
        (static_cast<std::size_t>(t) * B + seq) * DK;
    k_step[row] = keys[input_base + row];
    q_step[row] = queries[input_base + row];
  }
  if (row < DV) {
    const std::size_t value_base =
        (static_cast<std::size_t>(t) * B + seq) * DV;
    v_step[row] = values[value_base + row];
  }
  if (row == 0) {
    const std::size_t gate_index = static_cast<std::size_t>(t) * B + seq;
    alpha_step = alpha[gate_index];
    beta_step = beta[gate_index];
  }
  __syncthreads();

  if (row < DV) {
    const std::size_t state_base =
        (static_cast<std::size_t>(seq) * DV + row) * DK;
    float row_state[DK];
#pragma unroll
    for (int j = 0; j < DK; ++j) row_state[j] = states[state_base + j];

    float delta_dot = 0.0f;
#pragma unroll
    for (int j = 0; j < DK; ++j) {
      const float decayed = alpha_step * row_state[j];
      delta_dot = fmaf(decayed, k_step[j], delta_dot);
    }
    const float error = v_step[row] - delta_dot;

    float output = 0.0f;
#pragma unroll
    for (int j = 0; j < DK; ++j) {
      const float decayed = alpha_step * row_state[j];
      row_state[j] = fmaf(beta_step * error, k_step[j], decayed);
      output = fmaf(row_state[j], q_step[j], output);
    }
    outputs[(static_cast<std::size_t>(t) * B + seq) * DV + row] = output;
#pragma unroll
    for (int j = 0; j < DK; ++j) states[state_base + j] = row_state[j];
  }
}

// Fused state-retention candidate: one CTA owns one sequence, one lane owns
// one complete contiguous d_k row, and all live steps execute in order. The
// row stays in a private array for the sequence and final state is written once.
// Inspect registers/spills and residency at every measured d_k; the compiler
// may place this array in local memory instead of registers.
template <int DV, int DK>
__global__ void recurrent_retain_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const float* __restrict__ queries,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const int* __restrict__ lengths,
    float* __restrict__ states,
    float* __restrict__ outputs,
    int T,
    int B) {
  const int seq = blockIdx.x;
  const int row = threadIdx.x;
  if (seq >= B) return;
  const int sequence_length = lengths[seq];
  if (sequence_length == 0) return;

  __shared__ float k_step[DK];
  __shared__ float q_step[DK];
  __shared__ float v_step[DV];
  __shared__ float alpha_step;
  __shared__ float beta_step;

  const std::size_t state_base =
      (static_cast<std::size_t>(seq) * DV + row) * DK;
  float row_state[DK];
  if (row < DV) {
#pragma unroll
    for (int j = 0; j < DK; ++j) row_state[j] = states[state_base + j];
  }

  for (int t = 0; t < sequence_length && t < T; ++t) {
    if (row < DK) {
      const std::size_t input_base =
          (static_cast<std::size_t>(t) * B + seq) * DK;
      k_step[row] = keys[input_base + row];
      q_step[row] = queries[input_base + row];
    }
    if (row < DV) {
      const std::size_t value_base =
          (static_cast<std::size_t>(t) * B + seq) * DV;
      v_step[row] = values[value_base + row];
    }
    if (row == 0) {
      const std::size_t gate_index = static_cast<std::size_t>(t) * B + seq;
      alpha_step = alpha[gate_index];
      beta_step = beta[gate_index];
    }
    __syncthreads();

    if (row < DV) {
      float delta_dot = 0.0f;
#pragma unroll
      for (int j = 0; j < DK; ++j) {
        const float decayed = alpha_step * row_state[j];
        delta_dot = fmaf(decayed, k_step[j], delta_dot);
      }
      const float error = v_step[row] - delta_dot;

      float output = 0.0f;
#pragma unroll
      for (int j = 0; j < DK; ++j) {
        const float decayed = alpha_step * row_state[j];
        row_state[j] = fmaf(beta_step * error, k_step[j], decayed);
        output = fmaf(row_state[j], q_step[j], output);
      }
      outputs[(static_cast<std::size_t>(t) * B + seq) * DV + row] = output;
    }

    // Wait until all rows finish consuming the current shared input vectors.
    __syncthreads();
  }

  if (row < DV) {
#pragma unroll
    for (int j = 0; j < DK; ++j) states[state_base + j] = row_state[j];
  }
}

template <int DV, int DK>
inline void launch_materialized_sequence(
    const float* keys, const float* values, const float* queries,
    const float* alpha, const float* beta, const int* lengths,
    float* states, float* outputs, int T, int B, cudaStream_t stream) {
  constexpr int kThreads = DV > DK ? DV : DK;
  for (int t = 0; t < T; ++t) {
    recurrent_step_kernel<DV, DK>
        <<<B, kThreads, 0, stream>>>(keys, values, queries, alpha, beta,
                                     lengths, states, outputs, t, T, B);
  }
}

template <int DV, int DK>
inline void launch_retained_sequence(
    const float* keys, const float* values, const float* queries,
    const float* alpha, const float* beta, const int* lengths,
    float* states, float* outputs, int T, int B, cudaStream_t stream) {
  constexpr int kThreads = DV > DK ? DV : DK;
  recurrent_retain_kernel<DV, DK>
      <<<B, kThreads, 0, stream>>>(keys, values, queries, alpha, beta,
                                  lengths, states, outputs, T, B);
}

}  // namespace kwv08
