"""Collect build and regression evidence for the top-level lc0 relocation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
from lc0_upstream import modified_sources

folder = ROOT / "artifacts/layout_review"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((folder / (name + ".json")).read_text(encoding="utf-8"))


def log(name):
    data = (folder / (name + ".log")).read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


variants = ("ny", "no_ny", "onnx_ny", "onnx_no_ny", "makebook_ny", "makebook_no_ny",
            "multiponder_ny", "multiponder_no_ny", "tensorrt_double", "tensorrt_debug")
builds = {}
for variant in variants:
    exe = ("dlshogi_onnxruntime" if variant.startswith("onnx") else
           "usi_multiponder" if variant.startswith("multiponder") else
           "dlshogi_tensorrt_double" if variant == "tensorrt_double" else "dlshogi_tensorrt")
    assert (exe + ".exe") in log("build_" + variant)
    builds[variant] = {ext + "_sha256": sha(folder / variant / (exe + "." + ext))
                       for ext in ("exe", "pdb")}
for config in ("Release", "Debug"):
    assert "classic adapter tests passed" in log("unit_" + config)
defaults = (("ny", "usi/x64/Release", "dlshogi_tensorrt"),
            ("tensorrt_debug", "usi/x64/Debug", "dlshogi_tensorrt"),
            ("tensorrt_double", "usi/x64/Release_WinTypeDouble", "dlshogi_tensorrt_double"),
            ("makebook_ny", "usi/x64/MakeBook", "dlshogi_tensorrt"),
            ("onnx_no_ny", "usi_onnxruntime/x64/Release", "dlshogi_onnxruntime"),
            ("multiponder_no_ny", "usi_multiponder/x64/Release", "usi_multiponder"))
for variant, destination, exe in defaults:
    for ext in ("exe", "pdb"):
        assert sha(ROOT / destination / (exe + "." + ext)) == builds[variant][ext + "_sha256"]

before = read("before")
changed = [p for p, digest in before.items() if sha(ROOT / p) != digest]
assert set(changed) == {"lc0/CHANGES.txt", "lc0_shogi/classic.mk"}
assert not any((ROOT / p).exists() for p in
               ("usi/lc0", "usi/lc0_shogi", "usi/lc0_search.vcxproj"))
upstream = modified_sources(ROOT)
assert len(upstream) == 6
equivalence = {v: read("equivalence_" + v) for v in ("ny", "no_ny")}
for variant, result in equivalence.items():
    assert result["complete"] and result["exact_nn_counts_required"]
    assert len(result["records"]) == 30 and result["transposed_cache_policy_equal"]
    assert result["old_sha256"] == sha(ROOT / "artifacts/reuse_fix" / variant / "dlshogi_tensorrt.exe")
    assert result["new_sha256"] == builds[variant]["exe_sha256"]
    for key in ("bestmove", "pv", "nn_count", "nn_batch"):
        field = key + "_difference_samples"
        assert result[field] == 0, (variant, field)
ready = {v: read("ready_" + v) for v in variants[:4]}
for variant, result in ready.items():
    assert result["complete"] and result["engine_sha256"] == builds[variant]["exe_sha256"]
    for key, value in result["expected"].items():
        actual = result["actual"][key]
        assert (math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-6)
                if isinstance(value, float) else actual == value), (variant, key)
assert all(row["rejected"] for row in read("boundary_negative"))
for name in ("make_usi", "make_onnx", "make_multiponder"):
    text = log(name)
    assert "../lc0/src/search/classic/search.cc" in text
    assert "../lc0_shogi/backend.cpp" in text

report = dict(complete=True, verified_at=datetime.now(timezone.utc).isoformat(),
              builds=builds, unit_configurations=["Release", "Debug"],
              moved_file_count=len(before), implementation_byte_identical=True,
              moved_metadata_differences=changed, lc0_upstream_modified_files=upstream,
              source_audit=read("source_audit"), boundary_negative=read("boundary_negative"),
              equivalence={v: {k: x for k, x in r.items() if k != "records"}
                           for v, r in equivalence.items()},
              ready_settings={v: r["checks"] for v, r in ready.items()},
              make_validation="three dry runs; Linux compilation not rerun",
              default_executables_and_symbols_updated=True,
              search_logic_changed=False)
(ROOT / "test/layout_review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("layout review passed: 10 builds, Release/Debug units, 60 exact comparisons, four readiness checks")
