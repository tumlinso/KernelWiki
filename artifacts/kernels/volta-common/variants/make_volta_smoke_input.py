#!/usr/bin/env python3
"""Write the fixed little-endian FP16 matrices consumed by volta_smoke.cu."""
import hashlib
import json
import struct
import sys
from pathlib import Path

side = 16
values = []
for matrix, mul, offset, modulus in ((0, 5, 3, 17), (1, 7, 1, 19)):
    for i in range(side * side):
        n = (i * mul + offset) % modulus - modulus // 2
        values.append(struct.pack("<e", n / 16.0))
payload = b"".join(values)
if len(payload) != 2 * side * side * 2:
    raise SystemExit("unexpected input size")
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(payload)
print(json.dumps({"path": str(output), "bytes": len(payload),
                  "sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))
