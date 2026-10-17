"""Deterministic binary represented-input and FP64-reference bundle for KWV-07."""
from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from pathlib import Path
from typing import Any

from contract import (
    Assignment,
    GroupedFixture,
    decode_w4a16,
    check_w4a16_fp32_simulation,
    gamma_bound,
    grouped_gated_admission_bound,
    fp16,
    f32,
    gated_oracle,
    pack_signed_nibbles,
    stable_grouping,
)

INPUT_MAGIC = b'KWV07I01'
ORACLE_MAGIC = b'KWV07R01'
HEADER = struct.Struct('<8s8I')
KIND_GROUPED = 1
KIND_W4A16 = 2
GROUPED_CASES = (
    ('tiny_balanced', 8, 64, 32, 4, (2, 2, 2, 2)),
    ('tiny_skewed', 8, 64, 32, 4, (1, 0, 0, 7)),
    ('moderate_balanced', 32, 128, 64, 4, (8, 8, 8, 8)),
    ('moderate_skewed', 32, 128, 64, 4, (1, 0, 0, 31)),
    ('weighted_balanced', 128, 512, 256, 8, (16, 16, 16, 16, 16, 16, 16, 16)),
    ('weighted_skewed', 128, 512, 256, 8, (1, 0, 0, 0, 0, 0, 0, 127)),
)
W4_CASES = (
    ('low_m', 1, 257, 63, 128, 'tail_adversarial'),
    ('moderate_m', 16, 257, 63, 128, 'tail_adversarial'),
    ('weighted_low_m', 1, 4096, 1024, 128, 'shared_large_weights'),
    ('weighted_moderate_m', 16, 4096, 1024, 128, 'shared_large_weights'),
)
SCALE_CHOICES = (2.0 ** -14, 0.020010920241475105, 1.0, 8000.0,
                 2.0 ** -13, 0.5, 4000.0, 2.0)
MAX_RECORD_BYTES = 80 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _pack_halves(values: list[float] | tuple[float, ...]) -> bytes:
    return b''.join(struct.pack('<e', value) for value in values)


def _unpack_halves(data: bytes, count: int) -> tuple[float, ...]:
    if len(data) != count * 2:
        raise ValueError('FP16 buffer length does not match element count')
    return tuple(struct.unpack_from('<e', data, index * 2)[0] for index in range(count))


def _pack_ints(values: list[int] | tuple[int, ...]) -> bytes:
    if not values:
        return b''
    return struct.pack('<' + 'i' * len(values), *values)


def _unpack_ints(data: bytes, count: int) -> tuple[int, ...]:
    if len(data) != count * 4:
        raise ValueError('int32 buffer length does not match element count')
    if count == 0:
        return ()
    return struct.unpack('<' + 'i' * count, data)


def _pack_floats(values: list[float] | tuple[float, ...]) -> bytes:
    if not values:
        return b''
    return struct.pack('<' + 'f' * len(values), *values)


def _unpack_floats(data: bytes, count: int) -> tuple[float, ...]:
    if len(data) != count * 4:
        raise ValueError('FP32 buffer length does not match element count')
    if count == 0:
        return ()
    return struct.unpack('<' + 'f' * count, data)


def _pack_doubles(values: list[float] | tuple[float, ...]) -> bytes:
    if not values:
        return b''
    return struct.pack('<' + 'd' * len(values), *values)


def _unpack_doubles(data: bytes, count: int) -> tuple[float, ...]:
    if len(data) != count * 8:
        raise ValueError('FP64 buffer length does not match element count')
    if count == 0:
        return ()
    return struct.unpack('<' + 'd' * count, data)


def _grouped_fixture(case: tuple[str, int, int, int, int, tuple[int, ...]], seed: int) -> GroupedFixture:
    _case_id, assignments_count, k, n, experts, counts = case
    rng = random.Random(seed)
    rows = max(4, min(assignments_count, assignments_count // 2))
    x = tuple(fp16(rng.randint(-32, 32) / 64.0) for _ in range(rows * k))
    up = tuple(fp16(rng.randint(-32, 32) / 64.0) for _ in range(experts * k * n))
    gate = tuple(fp16(rng.randint(-32, 32) / 64.0) for _ in range(experts * k * n))
    records: list[tuple[int, int]] = []
    for expert, count in enumerate(counts):
        expert_rows = []
        for i in range(count):
            row = (expert * 3 + i * 5) % rows
            expert_rows.append(row)
        if count >= 2:
            expert_rows[1] = expert_rows[0]  # explicit repeated (row, expert)
        records.extend((row, expert) for row in expert_rows)
    if len(records) != assignments_count:
        raise ValueError('grouped case counts do not equal M')
    rng.shuffle(records)
    positions = list(range(assignments_count))
    rng.shuffle(positions)
    assignments = tuple(Assignment(row_id, expert_id, positions[i])
                        for i, (row_id, expert_id) in enumerate(records))
    fixture = GroupedFixture(rows, k, n, experts, x, up, gate, assignments)
    stable_grouping(fixture)
    return fixture


def _input_grouped(fixture: GroupedFixture) -> bytes:
    rows, k, n, experts = fixture.rows, fixture.k, fixture.n, fixture.experts
    header = HEADER.pack(INPUT_MAGIC, KIND_GROUPED, rows, k, n, experts,
                         0, len(fixture.assignments), 0)
    triples = tuple(v for a in fixture.assignments
                    for v in (a.row_id, a.expert_id, a.output_position))
    return (header + _pack_halves(fixture.x16) + _pack_halves(fixture.up16) +
            _pack_halves(fixture.gate16) + _pack_ints(triples))


def _grouped_oracle(fixture: GroupedFixture) -> bytes:
    output_count = len(fixture.assignments) * fixture.n
    y = [0.0] * output_count
    up_values = [0.0] * output_count
    gate_values = [0.0] * output_count
    sum_abs_up = [0.0] * output_count
    sum_abs_gate = [0.0] * output_count
    for assignment in fixture.assignments:
        for col in range(fixture.n):
            up = gate = up_abs = gate_abs = 0.0
            for kk in range(fixture.k):
                x = float(fixture.x16[assignment.row_id * fixture.k + kk])
                wu = float(fixture.matrix_value(fixture.up16, assignment.expert_id, kk, col))
                wg = float(fixture.matrix_value(fixture.gate16, assignment.expert_id, kk, col))
                up += x * wu
                gate += x * wg
                up_abs += abs(x * wu)
                gate_abs += abs(x * wg)
            if gate >= 0.0:
                activation = gate / (1.0 + math.exp(-gate))
            else:
                exp_value = math.exp(gate)
                activation = gate * exp_value / (1.0 + exp_value)
            index = assignment.output_position * fixture.n + col
            up_values[index] = up
            gate_values[index] = gate
            y[index] = activation * up
            sum_abs_up[index] = up_abs
            sum_abs_gate[index] = gate_abs
    header = HEADER.pack(ORACLE_MAGIC, KIND_GROUPED, fixture.rows, fixture.k,
                         fixture.n, fixture.experts, 0, len(fixture.assignments),
                         output_count)
    payloads = (y, up_values, gate_values, sum_abs_up, sum_abs_gate)
    return header + b''.join(_pack_doubles(values) for values in payloads)



def _w4_inputs(m: int, k: int, n: int, seed: int, group_size: int) -> tuple[tuple[float, ...], bytes, tuple[float, ...]]:
    rng = random.Random(seed)
    activation = tuple(fp16(rng.randint(-64, 64) / 128.0) for _ in range(m * k))
    codes = [((index * 7 + 3) % 16) - 8 for index in range(k * n)]
    packed = pack_signed_nibbles(codes)
    groups = (k + group_size - 1) // group_size
    scales = tuple(f32(SCALE_CHOICES[(group * n + col + seed) % len(SCALE_CHOICES)])
                   for group in range(groups) for col in range(n))
    return activation, packed, scales


def _input_w4(m: int, k: int, n: int, group_size: int,
              activation: tuple[float, ...], packed: bytes,
              scales: tuple[float, ...]) -> bytes:
    scale_count = len(scales)
    header = HEADER.pack(INPUT_MAGIC, KIND_W4A16, m, k, n, 0,
                         group_size, 0, scale_count)
    return header + _pack_halves(activation) + packed + _pack_floats(scales)


def _w4_oracle(m: int, k: int, n: int, group_size: int,
               activation: tuple[float, ...], packed: bytes,
               scales: tuple[float, ...]) -> bytes:
    decoded = decode_w4a16(packed, scales, k, n, group_size)
    flattened: list[float] = []
    sum_abs = []
    for row in range(m):
        for col in range(n):
            dot = absolute_products = 0.0
            for kk in range(k):
                term = float(activation[row * k + kk]) * float(decoded[kk * n + col])
                dot += term
                absolute_products += abs(term)
            flattened.append(dot)
            sum_abs.append(absolute_products)
    header = HEADER.pack(ORACLE_MAGIC, KIND_W4A16, m, k, n, 0,
                         group_size, 0, len(flattened))
    return header + _pack_doubles(flattened) + _pack_doubles(sum_abs)


def _read_header(data: bytes, expected_magic: bytes) -> tuple[int, ...]:
    if len(data) < HEADER.size:
        raise ValueError('binary record is shorter than its fixed header')
    magic, *values = HEADER.unpack_from(data)
    if magic != expected_magic:
        raise ValueError('binary record has wrong magic')
    return tuple(values)


def read_input(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    kind, rows, k, n, experts, group_size, assignment_count, scale_count = _read_header(data, INPUT_MAGIC)
    offset = HEADER.size
    if kind == KIND_GROUPED:
        x_count, weight_count = rows * k, experts * k * n
        x = _unpack_halves(data[offset:offset + x_count * 2], x_count)
        offset += x_count * 2
        up = _unpack_halves(data[offset:offset + weight_count * 2], weight_count)
        offset += weight_count * 2
        gate = _unpack_halves(data[offset:offset + weight_count * 2], weight_count)
        offset += weight_count * 2
        triple_count = assignment_count * 3
        triples = _unpack_ints(data[offset:offset + triple_count * 4], triple_count)
        offset += triple_count * 4
        if offset != len(data) or group_size != 0 or scale_count != 0:
            raise ValueError('grouped input has trailing bytes or inconsistent header')
        assignments = tuple(Assignment(*triples[i:i + 3])
                            for i in range(0, len(triples), 3))
        fixture = GroupedFixture(rows, k, n, experts, x, up, gate, assignments)
        stable_grouping(fixture)
        return {'kind': kind, 'rows': rows, 'k': k, 'n': n, 'experts': experts,
                'assignments': assignments, 'x16': x, 'up16': up, 'gate16': gate}
    if kind == KIND_W4A16:
        m = rows
        if experts != 0 or assignment_count != 0 or group_size <= 0:
            raise ValueError('W4A16 input header has invalid fields')
        activation_count = m * k
        activation = _unpack_halves(data[offset:offset + activation_count * 2], activation_count)
        offset += activation_count * 2
        packed_count = (k * n + 1) // 2
        packed = data[offset:offset + packed_count]
        if len(packed) != packed_count:
            raise ValueError('truncated packed W4A16 weights')
        offset += packed_count
        scales = _unpack_floats(data[offset:offset + scale_count * 4], scale_count)
        offset += scale_count * 4
        if offset != len(data):
            raise ValueError('W4A16 input has trailing bytes')
        if len(scales) != ((k + group_size - 1) // group_size) * n:
            raise ValueError('W4A16 scale count does not match K-group shape')
        # q is bounded by eight in magnitude. This validates every possible
        # nibble without materializing millions of decoded Python objects.
        if any(not math.isfinite(scale) or scale <= 0.0 or 8.0 * scale > 65504.0
               for scale in scales):
            raise ValueError('W4A16 scale violates positive finite FP16 decode bounds')
        return {'kind': kind, 'm': m, 'k': k, 'n': n, 'group_size': group_size,
                'activation16': activation, 'packed_codes': packed, 'scales': scales,
                'packed_bytes': len(packed), 'weight_elements': k * n}
    raise ValueError(f'unknown KWV-07 input kind {kind}')


def read_oracle(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    kind, rows, k, n, experts, group_size, assignment_count, output_count = _read_header(data, ORACLE_MAGIC)
    offset = HEADER.size
    if kind == KIND_GROUPED:
        expected_count = assignment_count * n
        if output_count != expected_count:
            raise ValueError('grouped oracle output count mismatch')
        y = _unpack_doubles(data[offset:offset + output_count * 8], output_count)
        offset += output_count * 8
        up_reference = _unpack_doubles(data[offset:offset + output_count * 8], output_count)
        offset += output_count * 8
        gate_reference = _unpack_doubles(data[offset:offset + output_count * 8], output_count)
        offset += output_count * 8
        up = _unpack_doubles(data[offset:offset + output_count * 8], output_count)
        offset += output_count * 8
        gate = _unpack_doubles(data[offset:offset + output_count * 8], output_count)
        offset += output_count * 8
        if offset != len(data) or group_size != 0:
            raise ValueError('grouped oracle has trailing bytes or inconsistent header')
        return {'kind': kind, 'rows': rows, 'k': k, 'n': n, 'experts': experts,
                'assignment_count': assignment_count, 'y64': y,
                'up_reference': up_reference, 'gate_reference': gate_reference,
                'sum_abs_up': up, 'sum_abs_gate': gate}
    if kind == KIND_W4A16:
        if experts != 0 or assignment_count != 0 or output_count != rows * n:
            raise ValueError('W4A16 oracle header has invalid fields')
        y = _unpack_doubles(data[offset:offset + output_count * 8], output_count)
        offset += output_count * 8
        sum_abs = _unpack_doubles(data[offset:offset + output_count * 8], output_count)
        offset += output_count * 8
        if offset != len(data):
            raise ValueError('W4A16 oracle has trailing bytes')
        return {'kind': kind, 'm': rows, 'k': k, 'n': n,
                'group_size': group_size, 'y64': y, 'sum_abs_products': sum_abs}
    raise ValueError(f'unknown KWV-07 oracle kind {kind}')


def _write_case(root: Path, case_id: str, input_data: bytes, oracle_data: bytes,
                contract: dict[str, Any]) -> dict[str, Any]:
    if len(input_data) >= MAX_RECORD_BYTES or len(oracle_data) >= MAX_RECORD_BYTES:
        raise ValueError(f'{case_id}: represented input/reference exceeds the 80 MiB record limit')
    input_path = root / 'inputs' / f'{case_id}.bin'
    oracle_path = root / 'oracles' / f'{case_id}.bin'
    contract_path = root / 'contracts' / f'{case_id}.json'
    input_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(input_data)
    oracle_path.write_bytes(oracle_data)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'id': case_id,
            'input': {'path': str(input_path.relative_to(root)), 'bytes': len(input_data), 'sha256': sha256_file(input_path)},
            'oracle': {'path': str(oracle_path.relative_to(root)), 'bytes': len(oracle_data), 'sha256': sha256_file(oracle_path)},
            'contract': {'path': str(contract_path.relative_to(root)), 'bytes': contract_path.stat().st_size, 'sha256': sha256_file(contract_path)}}


def generate_bundle(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(GROUPED_CASES):
        case_id, m, k, n, experts, counts = case
        seed = 0x07072026 if case_id in {'weighted_balanced', 'weighted_skewed'} else 0x700701 + index
        fixture = _grouped_fixture(case, seed)
        input_data = _input_grouped(fixture)
        oracle_data = _grouped_oracle(fixture)
        contract = {
            'kind': 'grouped_gated', 'operator': 'Y[pos] = silu(X[row] W_gate[expert]) * (X[row] W_up[expert])',
            'shape': {'source_rows': fixture.rows, 'assignments': m, 'k': k, 'n': n, 'experts': experts},
            'group_counts': list(counts), 'group_order': 'stable expert-major permutation of original assignment indices',
            'assignment_contract': 'row_id, expert_id, output_position; duplicate rows/experts retained and output positions unique',
            'numerics': {'input': 'represented IEEE binary16', 'projection_accumulator': 'FP32',
                         'projection_intermediate': 'FP32', 'silu': 'FP32 g/(1+expf(-g))',
                         'product_and_output': 'FP32', 'oracle': 'FP64 over exact represented input values',
                         'projection_bound': 'gamma_K * sum(abs(x16*w16)) per output',
                         'gated_output_bound': 'abs(silu(G64))*eU + (abs(U64)+eU)*(1+1/e)*eG + 8*2^-24*(1+abs(Y64)+abs(U64)+abs(G64))',
                         'exceptional_values': f'group fixture inputs and weights are finite with magnitude <=0.5; K<={k} bounds |gate| by {0.5 * 0.5 * k:g}'},
            'input_sha256': sha256_bytes(input_data), 'oracle_sha256': sha256_bytes(oracle_data)}
        cases.append(_write_case(root, case_id, input_data, oracle_data, contract))
    shared_weights: dict[str, tuple[bytes, tuple[float, ...]]] = {}
    for index, (case_id, m, k, n, group_size, weight_family) in enumerate(W4_CASES):
        activation, packed, scales = _w4_inputs(m, k, n, 0x70A701 + index, group_size)
        if weight_family in shared_weights:
            packed, scales = shared_weights[weight_family]
        else:
            shared_weights[weight_family] = (packed, scales)
        input_data = _input_w4(m, k, n, group_size, activation, packed, scales)
        oracle_data = _w4_oracle(m, k, n, group_size, activation, packed, scales)
        contract = {
            'kind': 'w4a16', 'operator': 'C[m,n] = sum_k A16[m,k] * round_fp16_rn(round_fp32(scale[group_k,n] * q[k,n]))',
            'shape': {'m': m, 'k': k, 'n': n, 'group_size': group_size},
            'format': {'codes': 'signed two-complement int4 [-8,7]',
                       'packing': 'row-major [K,N], even code low nibble, odd code high nibble; unused tail nibble ignored',
                       'scales': 'positive finite FP32 [ceil(K/group_size),N]',
                       'scale_domain_tested': [2.0 ** -14, 8000.0]},
            'numerics': {'activation': 'represented IEEE binary16', 'decoded_weight': 'FP16 RN after FP32 scale multiply',
                         'accumulator': 'FP32', 'output': 'FP32', 'oracle': 'FP64 over represented A16 and decoded W16',
                         'dot_bound': 'gamma_K * sum(abs(A16*decoded_W16))',
                         'exceptional_values': 'activation and decoded weights finite; scale*q must round to finite binary16; bounded fixture scale envelope is [2^-14,8000]'},
            'reuse_cases': [1, 4, 16] if m == 1 else [1, 4],
            'weight_family': weight_family,
            'input_sha256': sha256_bytes(input_data), 'oracle_sha256': sha256_bytes(oracle_data)}
        cases.append(_write_case(root, case_id, input_data, oracle_data, contract))

    manifest = {
        'format': 'KERNELWIKI-KWV07-BUNDLE/1',
        'binary_format': {'header': 'little-endian <8s8I, 40 bytes',
                          'input_magic': INPUT_MAGIC.decode(), 'oracle_magic': ORACLE_MAGIC.decode(),
                          'grouped_input_payload': 'FP16 X, FP16 W_up, FP16 W_gate, int32 assignments[row,expert,output_position]',
                          'grouped_oracle_payload': 'FP64 Y, FP64 U, FP64 G, FP64 sum_abs_up, FP64 sum_abs_gate',
                          'w4_input_payload': 'FP16 activations, packed signed nibbles, FP32 K-group scales',
                          'w4_oracle_payload': 'FP64 C, FP64 sum_abs_products'},
        'cases': cases,
    }
    manifest_path = root / 'bundle-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    # Keep two family records so evidence can represent one measured suite per
    # operator rather than multiplying every benchmark role by each fixture.
    from suite_records import generate_suite_records
    generate_suite_records(root, manifest)
    return manifest


def verify_bundle(root: Path) -> dict[str, Any]:
    manifest_path = root / 'bundle-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'KERNELWIKI-KWV07-BUNDLE/1':
        raise ValueError('unknown bundle manifest format')
    seen: set[str] = set()
    for case in manifest.get('cases', []):
        case_id = case.get('id')
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('case id is missing or duplicated')
        seen.add(case_id)
        input_path = root / case['input']['path']
        oracle_path = root / case['oracle']['path']
        contract_path = root / case['contract']['path']
        for path, record in ((input_path, case['input']), (oracle_path, case['oracle']),
                             (contract_path, case['contract'])):
            if path.stat().st_size != record.get('bytes') or sha256_file(path) != record.get('sha256'):
                raise ValueError(f'bundle SHA-256 mismatch: {path.name}')
        parsed_input = read_input(input_path)
        parsed_oracle = read_oracle(oracle_path)
        if parsed_input['kind'] != parsed_oracle['kind']:
            raise ValueError(f'input/oracle kind mismatch for {case_id}')
        # Family generation writes the FP64 CPU references once; its suite
        # wrappers bind their exact hashes and precheck the declared FP32
        # arithmetic. This verifier checks file identities and structure; it
        # deliberately does not regenerate large weighted FP64 products just
        # to re-establish identities already bound by SHA-256.
    from suite_records import verify_suite_records
    suite_hashes = verify_suite_records(root, manifest)
    return {'cases': len(seen), 'case_ids': sorted(seen), 'suite_hashes': suite_hashes}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('generate', 'verify'))
    parser.add_argument('bundle_dir', type=Path)
    args = parser.parse_args()
    if args.command == 'generate':
        result = generate_bundle(args.bundle_dir)
    else:
        result = verify_bundle(args.bundle_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
