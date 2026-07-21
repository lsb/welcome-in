"""Wire-size experiment: round an fp16 ONNX model's weights to fp8-e5m2 (or
e5mN) precision *stored as fp16*, so the low mantissa byte becomes zero and
gzip/zopfli halves the transfer — no runtime change, the file still runs as
plain fp16.

e5m2 is exactly the top byte of an fp16 (sign + 5 exp + 2 mantissa), so
"round to e5m2, keep fp16 container" = round each 16-bit word to the nearest
multiple of 0x100 in bit space (bit patterns are monotonic in magnitude, and
rounding carries propagate correctly through the exponent). Overflow past
0x7C00 (inf) clamps to 0x7B00 (e5m2 max, 57344); inf/NaN pass through.

Measured on the shipped MobileCLIP2-S0 vision fp16 (23.9 MB raw, 21.1 MB
pigz -11) against the sample-PNG tag harness — the collapse curve:

  mode   pigz-11   cos vs fp16   verdict
  e5m6   16.3 MB   >= 0.991      tags/margins identical — the free win
  e5m4   13.3 MB   ~0.92         edge: 2 tag flips, margins move ~0.01
  e5m3   12.1 MB   ~0.72         broken (false-positive gate, hat lost)
  e5m2   10.1 MB   ~0.2          destroyed
  e4m3    9.9 MB   ~0.05         destroyed HARDER than e5m2 despite more
                                 mantissa — the e4 exponent range (min
                                 subnormal 2^-9) flushes this net's many
                                 tiny BN-folded conv weights; range, not
                                 precision, is the binding constraint.

Usage: uv run --with onnx python webapp/scripts/squash_fp16.py <model.onnx> <mantissa_bits|e4m3> <out.onnx>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def squash_fp16_bits(u: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Round fp16 bit patterns (uint16) to `mantissa_bits` of mantissa."""
    drop = 10 - mantissa_bits
    if drop <= 0:
        return u
    step = 1 << drop
    half = step >> 1
    keep = np.uint16(~(step - 1) & 0xFFFF)
    sign = u & np.uint16(0x8000)
    mag = u & np.uint16(0x7FFF)
    special = mag >= 0x7C00  # inf/NaN untouched
    rounded = ((mag.astype(np.uint32) + half) & keep).astype(np.uint16)
    max_finite = np.uint16(0x7C00 - step)  # largest e5mN-representable
    rounded = np.where(rounded >= 0x7C00, max_finite, rounded)
    return np.where(special, u, sign | rounded)


def squash_e4m3(u: np.ndarray) -> np.ndarray:
    """Round fp16 bit patterns through true fp8-e4m3 (3-bit mantissa AND the
    e4 exponent range: clamp to ±448, subnormals at multiples of 2^-9), back
    to fp16. This is what an fp16->fp8e4m3->fp16 cast pair would produce."""
    x = u.view("<f2").astype(np.float32)
    sign = np.sign(x)
    a = np.minimum(np.abs(x), 448.0)
    mant, exp = np.frexp(a)  # a = mant * 2^exp, mant in [0.5, 1)
    normal = np.ldexp(np.round(mant * 16.0) / 16.0, exp)  # 4 significand bits
    sub = np.round(a * 512.0) / 512.0  # subnormal grid 2^-9
    q = np.where(a >= 2.0**-6, normal, sub)
    return (sign * q).astype("<f2").view("<u2")


def main() -> None:
    import onnx

    src, mode, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    m = onnx.load(str(src))
    n_tensors = n_params = 0
    for init in m.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT16 or not init.raw_data:
            continue
        u = np.frombuffer(init.raw_data, "<u2")
        out = squash_e4m3(u) if mode == "e4m3" else squash_fp16_bits(u, int(mode))
        init.raw_data = out.astype("<u2").tobytes()
        n_tensors += 1
        n_params += u.size
    onnx.save(m, str(dst))
    label = mode if mode == "e4m3" else f"e5m{mode}"
    print(f"squashed {n_tensors} fp16 tensors ({n_params/1e6:.1f}M params) "
          f"to {label}: {dst} ({dst.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
