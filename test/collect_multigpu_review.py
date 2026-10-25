"""Collect single-GPU concurrency evidence; do not infer multi-GPU scaling."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lc0_upstream import modified_sources
from evaluate_classic import ROOT, file_sha256

folder = ROOT / "artifacts/multigpu_review"
output = ROOT / "test/multigpu_review.json"
output.unlink(missing_ok=True)
def load(name):
    return json.loads((folder / (name + ".json")).read_text(encoding="utf-8"))
def log(path):
    raw = path.read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")

summary = {"complete": True, "verified_at": datetime.now(timezone.utc).isoformat(),
           "physical_gpu_count": 1, "gpu": "RTX 4090", "nn": {}, "equivalence": {},
           "book": {}, "matrix": {}, "protocol": {}, "real_backend": {}, "builds": {}}
for variant in ("ny", "no_ny", "onnx_ny", "onnx_no_ny", "makebook_ny", "makebook_no_ny"):
    nn = load("nn_" + variant)
    assert nn["complete"] and len(nn["records"]) == 100 and not nn["requires_investigation"]
    assert nn["new_sha256"] == file_sha256(nn["new"])
    assert nn["model_sha256"] == file_sha256(nn["model"])
    assert all(r["value_error"] <= 1e-3 and r["policy_error"] <= 1e-3 and r["one_visit_cp_error"] <= 1
               for r in nn["records"])
    summary["nn"][variant] = dict(engine_sha256=nn["new_sha256"], model_sha256=nn["model_sha256"], positions=100)
for model in ("ny", "no_ny"):
    result = load("equivalence_" + model)
    assert result["complete"] and len(result["records"]) == 30 and not result["observe_only"]
    assert result["bestmove_difference_samples"] == result["pv_difference_samples"] == 0
    assert result["transposed_cache_policy_equal"]
    assert result["old_sha256"] == file_sha256(ROOT / f"artifacts/reserve_review/{model}/dlshogi_tensorrt.exe")
    assert result["new_sha256"] == summary["nn"][model]["engine_sha256"]
    summary["equivalence"][model] = {k:v for k,v in result.items() if k != "records"}
    book = load("book_" + model)
    assert book["readback"] and book["policy_blend"] and book["makebook_policy_blend"]
    assert book["reader_sha256"] == summary["nn"][model]["engine_sha256"]
    assert book["engine_sha256"] == summary["nn"]["makebook_" + model]["engine_sha256"]
    summary["book"][model] = book
for variant in ("ny", "no_ny", "onnx_ny", "onnx_no_ny"):
    matrix = load("matrix_" + variant)
    assert matrix["complete"] and matrix["gpu_ids"] == [0] and len(matrix["results"]) == 8
    assert matrix["positions"] == 6 and matrix["nodes_budget"] == 4000 and matrix["repeats"] == 1
    assert matrix["model_sha256"] == summary["nn"][variant]["model_sha256"]
    assert {(r["threads_per_gpu"], r["task_workers"], r["max_concurrent_searchers"]) for r in matrix["results"]} == {
        (t,h,s) for t in (1,2) for h in (0,-1) for s in (1,0)}
    assert all(r["engine_sha256"] == summary["nn"][variant]["engine_sha256"] and len(r["records"]) == 6
               for r in matrix["results"])
    summary["matrix"][variant] = {**{k:v for k,v in matrix.items() if k not in ("results", "legacy")},
                                 "results": [{k:v for k,v in r.items() if k != "records"} for r in matrix["results"]]}
    protocol = load("protocol_" + variant)
    assert protocol["engine_sha256"] == summary["nn"][variant]["engine_sha256"]
    assert protocol["threads"] == 2 and protocol["task_workers"] == -1
    assert protocol["max_concurrent_searchers"] == 0 and len(protocol["passed"]) >= 21
    summary["protocol"][variant] = protocol
for variant in ("backend_trt", "backend_onnx", "backend_debug"):
    assert "backend concurrent AddInput passed; writers=8 rounds=8 queued=96 capacity=160 gpus=1" in log(folder / variant / "run.log")
    summary["real_backend"][variant] = dict(engine_sha256=file_sha256(folder / variant / "backend_concurrency.exe"),
                                          writers=8, rounds=8, requests_per_round=96, overflow_checked=True)
for config in ("Release", "Debug"):
    assert "classic adapter tests passed" in log(folder / f"unit_{config}.log")
summary["unit_tests"] = ["Release", "Debug"]
debug = load("debug_counters")
assert debug["complete"] and len(debug["checks"]) == 3
assert debug["engine_sha256"] == file_sha256(folder / "tensorrt_debug/dlshogi_tensorrt.exe")
summary["debug_counters"] = debug
for variant in ("ny", "no_ny", "onnx_ny", "onnx_no_ny", "makebook_ny", "makebook_no_ny", "multiponder_ny", "multiponder_no_ny", "tensorrt_double", "tensorrt_debug"):
    exe = "usi_multiponder.exe" if variant.startswith("multiponder") else ("dlshogi_onnxruntime.exe" if variant.startswith("onnx") else ("dlshogi_tensorrt_double.exe" if variant == "tensorrt_double" else "dlshogi_tensorrt.exe"))
    summary["builds"][variant] = file_sha256(folder / variant / exe)
summary["upstream_modified_files"] = modified_sources(ROOT)
assert len(summary["upstream_modified_files"]) == 6
summary["default_options"] = dict(TaskWorkers=0, MaxConcurrentSearchers=1, UCT_Threads=2, UCT_Threads2_through_8=0)
summary["limitations"] = ["No multi-GPU utilization/scaling measurement; demux/mux remain unimplemented.",
                          "Small single-GPU matrix is functional validation, not a performance or Elo claim."]
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("Passed: NN 600 positions; matrix 192 searches; helper-enabled protocol 84 checks; physical GPUs 1.")
