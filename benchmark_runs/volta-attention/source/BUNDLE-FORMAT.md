# Attention bundle format v1

All integers and floating-point values are little-endian. Offsets are byte
offsets from the beginning of each file. Both files begin with a 16-byte
header: 8-byte magic (`KWVATTN1` or `KWVREF01`), uint32 version `1`, uint32
case count. Fixed-size records immediately follow the header; each record's
payload arrays follow the complete record table. `attention_contract.json`
binds the exact input and reference bundle hashes.

## `KWVATTN1`

Each 200-byte case record has this order:

1. ASCII case identifier, NUL-padded to 64 bytes.
2. Ten uint32 values: `B,Hq,Hkv,Sq,Sk,D,flags,block_size,max_selected,reserved`.
3. Twelve uint64 values in pairs: Q offset/FP16 element count, K offset/count,
   V offset/count, boolean-mask offset/byte count, selected-ID offset/int32
   element count, selected-count offset/int32 element count.

Flags use bit 0 for right-aligned causal masking, bit 1 for supplied selected
blocks and bit 2 for a per-query boolean mask. Other bits are reserved. Q is
row-major `[B,Hq,Sq,D]`; K and V are row-major `[B,Hkv,Sk,D]`, stored as raw
IEEE binary16 values. Boolean mask bytes are `0` (allowed) or `1` (masked),
row-major `[B,Hq,Sq,Sk]`. Selected IDs and counts are signed int32 arrays;
each flattened `[B,Hq,Sq]` row owns `max_selected` ID slots and one count.
Unused slots are `-1`. IDs are unique within a row and fall in
`[0,ceil(Sk/block_size))`; zero counts represent an empty selection.

## `KWVREF01`

Each 128-byte case record contains the 64-byte case identifier, eight uint32
values `B,Hq,Hkv,Sq,Sk,D,reserved,reserved`, then output offset/count and row
stats offset/count as uint64 pairs. Output values are IEEE binary64, flattened
row-major `[B,Hq,Sq,D]`. Each 32-byte row record contains binary64
`max_score,normalizer,logsumexp` and uint64 `valid_keys`. Empty rows encode
`max_score=-inf`, `normalizer=0`, `logsumexp=-inf`, `valid_keys=0`, with an
exact zero output vector.

The format is intentionally simple for mmap/host readers and a native runner;
it is not a model-serving interchange format. Version or semantics changes
require a new format version and regenerated, re-hashed artifacts.
