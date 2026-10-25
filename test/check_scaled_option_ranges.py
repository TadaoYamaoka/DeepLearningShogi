"""Check integer USI bounds/scales against pinned upstream FloatOption ranges."""
from decimal import Decimal
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
source = (root / "lc0_shogi/classic_options.cpp").read_text(encoding="utf-8")
upstream = "\n".join((root / "lc0/src" / path).read_text(encoding="utf-8") for path in
                     ("search/classic/params.cc", "neural/shared_params.cc"))
ranges = {name.strip(): (Decimal(low.strip().removesuffix("f")), Decimal(high.strip().removesuffix("f")))
          for name, low, high in re.findall(r"Add<FloatOption>\((\w+),\s*([^,]+),\s*([^)]+)\)", upstream)}
count = 0
for name, initial, scale, minimum, maximum in re.findall(
        r'&(?:P|SharedBackendParams)::(\w+),\s*"([^"]+)",\s*(\d+),\s*(-?\d+),\s*(-?\d+)', source):
    scale = int(scale)
    if not scale:
        continue
    low, high = ranges[name]
    assert low * scale == int(minimum) and high * scale == int(maximum), name
    assert (Decimal(initial) * scale).to_integral_value() == Decimal(initial) * scale, name
    assert low <= Decimal(initial) <= high
    count += 1
assert count == 9
print("9 integer USI scales/ranges match pinned lc0 FloatOption definitions")
