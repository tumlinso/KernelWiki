"""Independent CPU contracts for the KWV-07 scratch preparation.

W4A16 wire format: signed two's-complement 4-bit q in [-8, 7], row-major
[K, N] code order, even code in the low nibble, odd code in the high nibble;
K-axis group scales are FP32 [ceil(K/group_size), N].  Decode each code by
FP32 multiply followed by round-to-nearest-even binary16 conversion.  GEMM
activations are represented binary16; products accumulate in FP32 and output
FP32.  These routines never consult a device result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence

DEFAULT_GROUP_SIZE = 128


def f32(value: float) -> float:
    """Round once to IEEE binary32, rejecting overflow/non-finite values."""
    try:
        result = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    except OverflowError as exc:
        raise ValueError('value is outside finite FP32 range') from exc
    if not math.isfinite(result):
        raise ValueError('FP32 value must be finite')
    return result


def fp16(value: float) -> float:
    """Round to IEEE binary16, nearest-even, then return its exact real value."""
    try:
        result = struct.unpack('<e', struct.pack('<e', float(value)))[0]
    except OverflowError as exc:
        raise ValueError('decoded value is outside finite FP16 range') from exc
    if not math.isfinite(result):
        raise ValueError('FP16 value must be finite')
    return result


def fp16_bits(value: float) -> int:
    return struct.unpack('<H', struct.pack('<e', fp16(value)))[0]


def from_fp16_bits(bits: int) -> float:
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def pack_signed_nibbles(codes: Iterable[int]) -> bytes:
    values = list(codes)
    packed = bytearray((len(values) + 1) // 2)
    for index, code in enumerate(values):
        if isinstance(code, bool) or not isinstance(code, int) or not -8 <= code <= 7:
            raise ValueError(f'code[{index}] must be an integer in [-8, 7]')
        nibble = code & 0x0f
        if index & 1:
            packed[index // 2] |= nibble << 4
        else:
            packed[index // 2] |= nibble
    return bytes(packed)


def unpack_signed_nibbles(packed: bytes, logical_count: int) -> list[int]:
    if isinstance(logical_count, bool) or not isinstance(logical_count, int) or logical_count < 0:
        raise ValueError('logical_count must be a non-negative integer')
    if len(packed) != (logical_count + 1) // 2:
        raise ValueError('packed byte count does not match logical_count')
    result = []
    for index in range(logical_count):
        byte = packed[index // 2]
        nibble = (byte >> (4 if index & 1 else 0)) & 0x0f
        result.append(nibble if nibble < 8 else nibble - 16)
    return result


def _scale_array(scales: Sequence[float], expected: int) -> list[float]:
    if len(scales) != expected:
        raise ValueError(f'expected {expected} FP32 scales, received {len(scales)}')
    result = [f32(value) for value in scales]
    if any(value <= 0.0 for value in result):
        raise ValueError('scales must be finite and positive')
    return result


def decode_w4a16(packed: bytes, scales: Sequence[float], k: int, n: int,
                  group_size: int = DEFAULT_GROUP_SIZE) -> list[float]:
    """Return exact represented FP16 weights in row-major [K, N] order."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (k, n, group_size)):
        raise ValueError('K, N, and group_size must be positive integers')
    count = k * n
    codes = unpack_signed_nibbles(packed, count)
    groups = (k + group_size - 1) // group_size
    represented_scales = _scale_array(scales, groups * n)
    decoded = []
    for index, code in enumerate(codes):
        row, col = divmod(index, n)
        scale = represented_scales[(row // group_size) * n + col]
        # CUDA contract: FP32 scale * exact integer q, then one FP16 RN-even cast.
        product = f32(scale * code)
        decoded.append(fp16(product))
    return decoded


def _half_matrix(values: Sequence[float], rows: int, cols: int, name: str) -> list[float]:
    if len(values) != rows * cols:
        raise ValueError(f'{name} extent does not match its shape')
    return [fp16(value) for value in values]


def w4a16_oracle(activation: Sequence[float], m: int, k: int, n: int,
                  packed_weights: bytes, scales: Sequence[float],
                  group_size: int = DEFAULT_GROUP_SIZE) -> list[list[float]]:
    """FP64 dot oracle over represented A16 and decoded W16 inputs."""
    a = _half_matrix(activation, m, k, 'activation')
    w = decode_w4a16(packed_weights, scales, k, n, group_size)
    return [[sum(float(a[row * k + kk]) * float(w[kk * n + col])
                 for kk in range(k))
              for col in range(n)] for row in range(m)]


@dataclass(frozen=True)
class Assignment:
    row_id: int
    expert_id: int
    output_position: int


@dataclass(frozen=True)
class GroupedFixture:
    rows: int
    k: int
    n: int
    experts: int
    x16: tuple[float, ...]
    up16: tuple[float, ...]  # [experts, K, N]
    gate16: tuple[float, ...]
    assignments: tuple[Assignment, ...]  # caller order; do not deduplicate/reorder

    def matrix_value(self, weight: Sequence[float], expert: int, kk: int, col: int) -> float:
        return weight[(expert * self.k + kk) * self.n + col]


def _small_half_sequence(count: int, salt: int) -> tuple[float, ...]:
    return tuple(fp16((((i * 17 + salt * 13) % 31) - 15) / 32.0)
                 for i in range(count))


def make_grouped_fixture(skewed: bool) -> GroupedFixture:
    rows, k, n, experts = 8, 8, 5, 6
    x = _small_half_sequence(rows * k, 2)
    up = _small_half_sequence(experts * k * n, 5)
    gate = _small_half_sequence(experts * k * n, 11)
    if skewed:
        # Four empty groups, one singleton and one heavy group.  The same
        # (row, expert) pair occurs twice at distinct output positions.
        records = (
            Assignment(3, 5, 6), Assignment(1, 5, 1), Assignment(3, 5, 4),
            Assignment(7, 2, 0), Assignment(3, 5, 5), Assignment(0, 5, 2),
            Assignment(2, 5, 3),
        )
    else:
        records = (
            Assignment(2, 4, 3), Assignment(0, 0, 0), Assignment(2, 4, 5),
            Assignment(7, 1, 2), Assignment(5, 3, 6), Assignment(4, 1, 1),
            Assignment(1, 3, 4),
        )
    return GroupedFixture(rows, k, n, experts, x, up, gate, records)


def stable_grouping(fixture: GroupedFixture) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return expert offsets and source assignment indices, preserving duplicates."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
           for v in (fixture.rows, fixture.k, fixture.n, fixture.experts)):
        raise ValueError('rows, K, N, and expert count must be positive integers')
    if len(fixture.x16) != fixture.rows * fixture.k or \
            len(fixture.up16) != fixture.experts * fixture.k * fixture.n or \
            len(fixture.gate16) != fixture.experts * fixture.k * fixture.n:
        raise ValueError('grouped fixture buffer extent does not match its declared shape')
    for values in (fixture.x16, fixture.up16, fixture.gate16):
        if any(not math.isfinite(v) or fp16(v) != v for v in values):
            raise ValueError('grouped inputs must be finite represented FP16 values')
    buckets: list[list[int]] = [[] for _ in range(fixture.experts)]
    for source_index, assignment in enumerate(fixture.assignments):
        if not 0 <= assignment.row_id < fixture.rows:
            raise ValueError(f'assignment[{source_index}] has invalid row_id')
        if not 0 <= assignment.expert_id < fixture.experts:
            raise ValueError(f'assignment[{source_index}] has invalid expert_id')
        if not 0 <= assignment.output_position < len(fixture.assignments):
            raise ValueError(f'assignment[{source_index}] has invalid output_position')
        buckets[assignment.expert_id].append(source_index)
    positions = [a.output_position for a in fixture.assignments]
    if sorted(positions) != list(range(len(positions))):
        raise ValueError('output_position values must be a permutation of result slots')
    offsets = [0]
    flat: list[int] = []
    for bucket in buckets:
        flat.extend(bucket)
        offsets.append(len(flat))
    if sorted(flat) != list(range(len(fixture.assignments))):
        raise AssertionError('grouping lost or duplicated assignment records')
    return tuple(offsets), tuple(flat)


def silu64(value: float) -> float:
    # Numerically stable real-valued SiLU reference.
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def gated_oracle(fixture: GroupedFixture) -> list[list[float]]:
    """FP64 projections and SiLU, scattered to explicit output positions."""
    stable_grouping(fixture)
    output = [[math.nan] * fixture.n for _ in fixture.assignments]
    for assignment in fixture.assignments:
        row, expert = assignment.row_id, assignment.expert_id
        values = []
        for col in range(fixture.n):
            up = 0.0
            gate = 0.0
            for kk in range(fixture.k):
                x = fixture.x16[row * fixture.k + kk]
                up += float(x) * float(fixture.matrix_value(fixture.up16, expert, kk, col))
                gate += float(x) * float(fixture.matrix_value(fixture.gate16, expert, kk, col))
            values.append(silu64(gate) * up)
        output[assignment.output_position] = values
    if any(any(not math.isfinite(x) for x in row) for row in output):
        raise AssertionError('fixture left an output position unassigned')
    return output


def gamma_bound(k: int, magnitude: float) -> float:
    ku = k * (2.0 ** -24)
    if k <= 0 or not ku < 1.0:
        raise ValueError('K invalidates the FP32 accumulation bound')
    return (ku / (1.0 - ku)) * magnitude


def grouped_gated_admission_bound(up64: float, gate64: float, y64: float,
                                  sum_abs_up: float, sum_abs_gate: float,
                                  k: int) -> float:
    error_up = gamma_bound(k, sum_abs_up)
    error_gate = gamma_bound(k, sum_abs_gate)
    silu = silu64(gate64)
    return (abs(silu) * error_up +
            (abs(up64) + error_up) * (1.0 + 1.0 / math.e) * error_gate +
            8.0 * (2.0 ** -24) * (1.0 + abs(y64) + abs(up64) + abs(gate64)))


def simulate_grouped_fp32(fixture: GroupedFixture) -> list[float]:
    """Sequential FP32-FMA policy simulation; it is not a GPU performance path."""
    stable_grouping(fixture)
    output = [0.0] * (len(fixture.assignments) * fixture.n)
    for assignment in fixture.assignments:
        for col in range(fixture.n):
            up = gate = 0.0
            for kk in range(fixture.k):
                x = fixture.x16[assignment.row_id * fixture.k + kk]
                wu = fixture.matrix_value(fixture.up16, assignment.expert_id, kk, col)
                wg = fixture.matrix_value(fixture.gate16, assignment.expert_id, kk, col)
                up = f32(float(x) * float(wu) + up)
                gate = f32(float(x) * float(wg) + gate)
            exp_value = math.exp(-gate)
            # CUDA expf returns +inf for sufficiently negative gates; then the
            # contracted direct quotient is signed zero. Avoid asking f32()
            # to accept that intermediate as a finite input.
            if exp_value > 3.4028234663852886e38:
                silu = 0.0
            else:
                silu = f32(gate / f32(1.0 + f32(exp_value)))
            value = f32(silu * up)
            output[assignment.output_position * fixture.n + col] = value
    return output


def check_grouped_fp32_simulation(fixture: GroupedFixture,
                                 oracle: Sequence[float],
                                 sum_abs_up: Sequence[float],
                                 sum_abs_gate: Sequence[float]) -> tuple[bool, float, float]:
    actual = simulate_grouped_fp32(fixture)
    reference = [value for row in gated_oracle(fixture) for value in row]
    if not (len(actual) == len(oracle) == len(sum_abs_up) == len(sum_abs_gate)):
        raise ValueError('grouped FP32 simulation/oracle extents differ')
    max_ratio = 0.0
    max_error = 0.0
    for index, value in enumerate(actual):
        assignment = index // fixture.n
        col = index % fixture.n
        record = next(a for a in fixture.assignments if a.output_position == assignment)
        up64 = 0.0
        gate64 = 0.0
        for kk in range(fixture.k):
            x = fixture.x16[record.row_id * fixture.k + kk]
            up64 += float(x) * float(fixture.matrix_value(fixture.up16, record.expert_id, kk, col))
            gate64 += float(x) * float(fixture.matrix_value(fixture.gate16, record.expert_id, kk, col))
        bound = grouped_gated_admission_bound(up64, gate64, reference[index],
                                              sum_abs_up[index], sum_abs_gate[index], fixture.k)
        error = abs(value - reference[index])
        max_error = max(max_error, error)
        ratio = 0.0 if bound == 0.0 and error == 0.0 else error / bound
        max_ratio = max(max_ratio, ratio)
    return max_ratio <= 1.0, max_error, max_ratio


def simulate_w4a16_fp32(activation: Sequence[float], m: int, k: int, n: int,
                        packed_weights: bytes, scales: Sequence[float],
                        group_size: int = DEFAULT_GROUP_SIZE) -> list[float]:
    a = _half_matrix(activation, m, k, 'activation')
    w = decode_w4a16(packed_weights, scales, k, n, group_size)
    output: list[float] = []
    for row in range(m):
        for col in range(n):
            acc = 0.0
            for kk in range(k):
                acc = f32(float(a[row * k + kk]) * float(w[kk * n + col]) + acc)
            output.append(acc)
    return output


def check_w4a16_fp32_simulation(activation: Sequence[float], m: int, k: int, n: int,
                                packed_weights: bytes, scales: Sequence[float],
                                reference: Sequence[float],
                                sum_abs_products: Sequence[float],
                                group_size: int = DEFAULT_GROUP_SIZE) -> tuple[bool, float, float]:
    actual = simulate_w4a16_fp32(activation, m, k, n, packed_weights, scales, group_size)
    if len(actual) != len(reference) or len(actual) != len(sum_abs_products):
        raise ValueError('W4A16 FP32 simulation/oracle extents differ')
    max_error = max_ratio = 0.0
    for value, expected, sum_abs in zip(actual, reference, sum_abs_products):
        bound = gamma_bound(k, sum_abs)
        error = abs(value - expected)
        max_error = max(max_error, error)
        ratio = 0.0 if bound == 0.0 and error == 0.0 else error / bound
        max_ratio = max(max_ratio, ratio)
    return max_ratio <= 1.0, max_error, max_ratio
