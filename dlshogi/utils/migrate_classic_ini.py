"""Migrate classic USI integer keys, preserving legacy model-ini bytes/values.

FPU coefficients are scaled only; the legacy FPU algorithm is not reproduced.
Use --output to review a separate file before replacing a model configuration.
"""
import argparse
from decimal import Decimal
from pathlib import Path
import sys


SCALES = {"CPuct":1000, "CPuctBase":1, "CPuctFactor":1000,
          "CPuctAtRoot":1000, "CPuctBaseAtRoot":1, "CPuctFactorAtRoot":1000,
          "FpuValue":1000, "FpuValueAtRoot":1000, "PolicyTemperature":1000}


def migrate(original, from_float_options=False):
    text = original.decode("utf-8-sig")
    values = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.strip().partition("=")
        if not separator or not key or not value:
            raise ValueError("Expected key=value: " + line)
        if key in values:
            raise ValueError("Duplicate ini key: " + key)
        values[key] = value

    def old(name, default, multiplier=1):
        value = Decimal(values.get(name, str(default))) * Decimal(str(multiplier))
        if not value.is_finite():
            raise ValueError("Nonfinite ini value: " + name)
        return format(value.normalize(), "f")

    added = {
        "CPuct": old("C_init", 144, "20"),
        "CPuctBase": old("C_base", 28288),
        "CPuctFactor": "2000",
        "CPuctAtRoot": old("C_init_root", 116, "20"),
        "CPuctBaseAtRoot": old("C_base_root", 25617),
        "CPuctFactorAtRoot": "2000",
        "RootHasOwnCpuctParams": "true",
        "FpuStrategy": "reduction",
        "FpuValue": old("C_fpu_reduction", 27, "20"),
        "FpuStrategyAtRoot": "reduction",
        "FpuValueAtRoot": old("C_fpu_reduction_root", 0, "20"),
        "PolicyTemperature": old("Softmax_Temperature", 174, "10"),
    }
    replacements = {}
    for key, scale in SCALES.items():
        if key not in values:
            continue
        value = Decimal(values[key]) * (scale if from_float_options else 1)
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError("Not representable as integer USI value: " + key +
                             "; use --from-float-options for previous float configurations")
        replacements[key] = str(int(value))
    if from_float_options:
        # Rewrite only new numeric keys. Legacy keys and their original bytes,
        # line endings and BOM stay untouched; no duplicate keys are appended.
        lines = []
        for line in original.splitlines(keepends=True):
            body = line.rstrip(b"\r\n")
            prefix, separator, value = body.partition(b"=")
            key = prefix.decode("utf-8-sig").strip()
            if key in replacements:
                line = prefix + separator + replacements[key].encode("ascii") + line[len(body):]
            lines.append(line)
        original = b"".join(lines)
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    additions = [f"{key}={value}".encode("ascii") for key, value in added.items() if key not in values]
    if not additions:
        return original
    separator = b"" if not original or original.endswith(b"\n") else newline
    return original + separator + newline.join(additions) + newline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-float-options", action="store_true",
                        help="Explicitly rescale the previous floating-point classic keys")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from input; review the converted file first")
    result = migrate(args.input.read_bytes(), args.from_float_options)
    args.output.write_bytes(result)
    print("Legacy keys preserved. FPU scaling does not reproduce legacy FPU behavior.", file=sys.stderr)


if __name__ == "__main__":
    main()
