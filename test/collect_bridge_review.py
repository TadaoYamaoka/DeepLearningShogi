"""Collect bridge-optimization regression evidence; no speed or Elo claims."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import ROOT, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=ROOT / "artifacts/bridge_review")
    parser.add_argument("--output", type=Path, default=ROOT / "test/bridge_review.json")
    args = parser.parse_args()
    args.output.unlink(missing_ok=True)
    reports = args.reports.resolve()
    def load(name):
        return json.loads((reports / (name + ".json")).read_text(encoding="utf-8"))
    verification = {}
    for variant in ("ny", "no_ny", "onnx_ny", "onnx_no_ny", "makebook_ny", "makebook_no_ny"):
        nn = load("nn_" + variant)
        assert nn["complete"] and len(nn["records"]) == 100 and not nn["requires_investigation"]
        assert nn["new_sha256"] == file_sha256(nn["new"])
        assert nn["model_sha256"] == file_sha256(nn["model"])
        assert all(r["value_error"] <= 1e-3 and r["policy_error"] <= 1e-3
                   and r["one_visit_cp_error"] <= 1 for r in nn["records"])
        entry = {"engine_sha256": nn["new_sha256"], "model_sha256": nn["model_sha256"],
                 "nn_positions": len(nn["records"]),
                 "max_value_error": max(r["value_error"] for r in nn["records"]),
                 "max_policy_error": max(r["policy_error"] for r in nn["records"])}
        if not variant.startswith("makebook"):
            protocol = load("protocol_" + variant)
            assert protocol["engine_sha256"] == nn["new_sha256"]
            assert protocol["threads"] == 2 and len(protocol["passed"]) >= 21
            entry["protocol_checks"] = len(protocol["passed"])
        verification[variant] = entry
    equivalence, books = {}, {}
    for model in ("ny", "no_ny"):
        result = load("equivalence_" + model)
        assert result["complete"] and len(result["records"]) == 30 and not result["observe_only"]
        assert result["new_sha256"] == verification[model]["engine_sha256"]
        assert result["old_sha256"] == file_sha256(ROOT / f"artifacts/inference_review/{model}/dlshogi_tensorrt.exe")
        assert result["transposed_cache_policy_equal"]
        assert result["bestmove_difference_samples"] == result["pv_difference_samples"] == 0
        equivalence[model] = {k: v for k, v in result.items() if k != "records"}
        book = load("book_" + model)
        assert book["readback"] and book["policy_blend"] and book["makebook_policy_blend"]
        assert book["reader_sha256"] == verification[model]["engine_sha256"]
        assert book["engine_sha256"] == verification["makebook_" + model]["engine_sha256"]
        books[model] = book
    debug = load("debug_counters")
    assert debug["complete"] and len(debug["checks"]) == 3
    assert debug["engine_sha256"] == file_sha256(reports / "tensorrt_debug/dlshogi_tensorrt.exe")
    multi = load("multiponder")
    assert multi["hits"] == multi["misses"] == 10 and multi["children_exited"] and multi["quit_during_ponder"]
    assert multi["auxiliary_sha256"] == verification["ny"]["engine_sha256"]
    assert multi["engine_sha256"] == file_sha256(reports / "multiponder_ny/usi_multiponder.exe")
    for config in ("Release", "Debug"):
        data = (reports / f"unit_{config}.log").read_bytes()
        text = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        assert "classic adapter tests passed" in text
    control = load("control_old_vs_old")
    assert control["complete"] and control["observe_only"] and control["old_sha256"] == control["new_sha256"]
    assert control["old_sha256"] == equivalence["ny"]["old_sha256"]
    smoke = {}
    for variant, executable in (("tensorrt_double", "dlshogi_tensorrt_double.exe"),
                                ("multiponder_no_ny", "usi_multiponder.exe")):
        data = (reports / f"smoke_{variant}.log").read_bytes()
        text = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        assert "bestmove " in text
        smoke[variant] = {"engine_sha256": file_sha256(reports / variant / executable), "threads": 2}
    result = {"complete": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_patch_sha256": file_sha256(reports / "input.patch"),
              "verification": verification, "equivalence": equivalence, "books": books,
              "debug_counter_checks": debug, "unit_configs": ["Release", "Debug"],
              "smoke": smoke,
              "multiponder": {k: v for k, v in multi.items() if k != "records"},
              "unchanged_binary_control": {k: v for k, v in control.items() if k != "records"},
              "notes": ["600 root NN comparisons include TensorRT, ONNX Runtime and MAKE_BOOK, each with NY and no-NY models.",
                        "Book-policy blending is checked separately with BOOK_POLICY enabled, including warmup-cache readback.",
                        "The real-GPU equivalence test requires identical bestmove/PV and records NN and visit counts; --require-exact-nn-counts enables stricter stub runs.",
                        "Initial strict runs observed stop-budget overshoot and a seven-evaluation difference at fixture 93 with identical bestmove/PV; observations are preserved in artifacts/bridge_review/initial_strict_observations.json.",
                        "Classic's MaybePrefetchIntoCache and GatherMinibatch inspect the asynchronous stop flag, so tail counts can depend on scheduling.",
                        "The review's approximately 64% callgrind reduction was not remeasured on this Windows host; no NPS or Elo improvement is asserted."]}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"nn_positions": 600, "equivalence": equivalence,
                      "control": result["unchanged_binary_control"]}, indent=2))


if __name__ == "__main__":
    main()
