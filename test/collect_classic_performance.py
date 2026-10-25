"""Collect the GPU performance follow-up without replacing earlier match evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import ROOT, FIXTURES, file_sha256


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_inference(reports):
    verification, measurements = {}, []
    for model in ("ny", "no_ny"):
        nn = load(reports / f"nn_{model}.json")
        protocol = load(reports / f"protocol_{model}.json")
        assert nn["complete"] and len(nn["records"]) == 100 and not nn["requires_investigation"]
        assert nn["new_sha256"] == protocol["engine_sha256"] == file_sha256(nn["new"])
        assert protocol["threads"] == 2 and len(protocol["passed"]) >= 21
        assert nn["model_sha256"] == protocol["model_sha256"] == file_sha256(nn["model"])
        verification[model] = {"engine_sha256": nn["new_sha256"],
                               "model_sha256": nn["model_sha256"],
                               "nn_positions": 100, "protocol_checks": len(protocol["passed"])}
        for pv in (0, 2):
            report = load(reports / f"{model}_pv{pv}.json")
            assert report["complete"] and report["threads"] == 2 and report["pv_threads"] == pv
            assert report["nodes_budget"] == 20000 and report["batch"] == 128 and report["repeats"] == 3
            assert report["positions_sha256"] == file_sha256(FIXTURES)
            assert report["new_sha256"] == nn["new_sha256"]
            assert report["model_sha256"] == nn["model_sha256"]
            assert report["old_sha256"] == file_sha256(report["old"])
            records = report["records"]
            assert len(records) == 90 and len({r["index"] for r in records}) == 30
            assert len({(r["index"], r["repeat"]) for r in records}) == 90
            assert set(Counter(r["index"] for r in records).values()) == {3}
            full = [r for r in records if all(r[label]["nodes"] >= 20000 for label in ("old", "new"))]
            assert len(full) == report["full_budget_positions"]
            for label in ("old", "new"):
                counters = load(reports / f"stats_{label}_{model}.json")
                assert counters["complete"] and counters["checks"]
                expected_hash = report[f"{label}_sha256"]
                assert counters["engine_sha256"] == expected_hash
                assert counters["model_sha256"] == nn["model_sha256"]
                verification[model].setdefault("counter_checks", {})[label] = counters["checks"]
                totals = report["inference"][label]
                assert totals["evaluations"] == sum(r[label]["nn_evaluations"] for r in full)
                assert totals["batches"] == sum(r[label]["nn_batches"] for r in full)
                assert totals["nodes"] == sum(r[label]["nodes"] for r in full)
                assert math.isclose(totals["seconds"], sum(r[label]["seconds"] for r in full))
                assert math.isclose(totals["evaluations_per_second"], totals["evaluations"] / totals["seconds"])
                assert math.isclose(totals["nodes_per_evaluation"], totals["nodes"] / totals["evaluations"])
                assert 1 <= totals["mean_batch_size"] <= 128
                assert all(r[label]["nn_evaluations"] >= r[label]["nn_batches"] >= 0 for r in records)
            inference = report["inference"]
            assert math.isclose(inference["aggregate_nps_ratio"],
                                inference["aggregate_rate_ratio"] * inference["visit_count_factor"])
            old, new = inference["old"], inference["new"]
            completion = {
                "old_seconds": old["seconds"], "new_seconds": new["seconds"],
                "time_ratio_new_over_old": new["seconds"] / old["seconds"],
                "old_actual_nodes": old["nodes"], "new_actual_nodes": new["nodes"],
                "old_nn_evaluations_per_node": old["evaluations"] / old["nodes"],
                "new_nn_evaluations_per_node": new["evaluations"] / new["nodes"],
                "nn_evaluations_per_node_ratio_new_over_old": 1 / inference["visit_count_factor"],
            }
            assert math.isclose(inference["aggregate_nps_ratio"] * completion["time_ratio_new_over_old"],
                                new["nodes"] / old["nodes"])
            measurements.append({"model": model, "pv_threads": pv,
                                 "old_sha256": report["old_sha256"], "new_sha256": report["new_sha256"],
                                 "full_budget_samples": len(full), "early_solved_samples": 90 - len(full),
                                 "median_old_nps": report["median_old_nps"],
                                 "median_new_nps": report["median_new_nps"],
                                 "median_paired_nps_ratio": report["median_position_ratio"],
                                 "completion": completion,
                                 "inference": report["inference"],
                                 "report": str((reports / f"{model}_pv{pv}.json").relative_to(ROOT))})
    additional = {}
    for model in ("ny", "no_ny"):
        path = reports / f"stats_onnx_{model}.json"
        if path.exists():
            counters = load(path)
            assert counters["complete"] and len(counters["checks"]) == 3
            assert counters["engine_sha256"] == file_sha256(reports / f"onnx_{model}/dlshogi_onnxruntime.exe")
            assert counters["model_sha256"] == verification[model]["model_sha256"]
            additional[f"onnx_{model}"] = counters
    return {"complete": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_and_driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
            "legacy_revision": "0cc16854474ab3c7c4f7be66c624f9f9001f7543",
            "legacy_instrumentation": "EvalNode completed-forward counters only; no root probe output",
            "backend": "TensorRT 8.6.1.6", "pinned_allocation": "cudaHostAllocPortable",
            "fixture_sha256": file_sha256(FIXTURES), "threads": 2, "nodes": 20000,
            "batch": 128, "positions": 30, "repeats": 3, "root_dfpn": 0,
            "primary_speed_metric": "inference.aggregate_nps_ratio",
            "primary_speed_definition": "(sum(new actual nodes)/sum(new search seconds))/(sum(old actual nodes)/sum(old search seconds)); greater than one is faster",
            "completion_time_definition": "completion.time_ratio_new_over_old compares summed wall time at the same nominal node budgets; greater than one is slower",
            "median_metrics_role": "Supplementary distribution diagnostics, not the primary speed conclusion",
            "verification": verification, "additional_counter_checks": additional, "measurements": measurements,
            "notes": ["Completed backend evaluations include prefetch; cache hits and terminal revisits are excluded.",
                      "Statistics are sampled after workers join; query time and initialization/warmup are excluded.",
                      "NN evaluations/second = sum(evaluations)/sum(search wall seconds); mean batch = sum(evaluations)/sum(forward calls).",
                      "Each condition retains both engine processes and NN caches across all repeats; subtree reuse is disabled.",
                      "Runs are sequential with alternating execution order; no concurrent builds or other engine tests.",
                      "NPS counts differ between search implementations, especially with PV dfpn; NPS ratios alone do not measure inference throughput.",
                      "Completion time is reported separately; comparable NN evaluations/second does not imply identical time to complete nominal node budgets.",
                      "Budget overshoot is retained in actual node totals, so the NPS ratio is not exactly the reciprocal of the completion-time ratio.",
                      "Cache-hit and prefetch contributions have not been isolated; a no-destination prefetch request does not prove that its cached result is never used later.",
                      "The collector reaggregates existing raw reports; its verification timestamp is not the benchmark execution time.",
                      "Prefetch and different leaf selection also affect NN throughput; these numbers do not measure Elo or useful search work."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Collect the completed-inference comparison")
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.reports = (args.reports or ROOT / ("artifacts/inference_review" if args.inference else "artifacts/performance_review")).resolve()
    args.output = args.output or ROOT / ("test/classic_inference_performance.json" if args.inference else "test/classic_performance.json")
    args.output.unlink(missing_ok=True)
    if args.inference:
        result = collect_inference(args.reports)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["measurements"], indent=2))
        return
    verification = {}
    for variant in ("ny", "after_no_ny", "onnx_ny", "onnx_no_ny"):
        nn = load(args.reports / f"nn_{variant}.json")
        protocol = load(args.reports / f"protocol_{variant}.json")
        assert nn["complete"] and len(nn["records"]) == 100
        assert not nn["requires_investigation"]
        assert all(r["value_error"] <= 1e-3 and r["policy_error"] <= 1e-3
                   and r["one_visit_cp_error"] <= 1 for r in nn["records"])
        assert protocol["threads"] == 2 and len(protocol["passed"]) >= 21
        assert nn["new_sha256"] == protocol["engine_sha256"] == file_sha256(nn["new"])
        assert nn["model_sha256"] == protocol["model_sha256"] == file_sha256(nn["model"])
        verification[variant] = {
            "engine": nn["new"], "sha256": nn["new_sha256"],
            "model_sha256": nn["model_sha256"], "nn_positions": 100,
            "protocol_checks": len(protocol["passed"]), "protocol_threads": 2,
            "max_value_error": max(r["value_error"] for r in nn["records"]),
            "max_policy_error": max(r["policy_error"] for r in nn["records"]),
        }
    measurements = []
    for model in ("ny", "no_ny"):
        expected = verification["ny" if model == "ny" else "after_no_ny"]
        for pv in (0, 2):
            path = args.reports / f"after_{model}_pv{pv}.json"
            report = load(path)
            assert report["complete"] and report["threads"] == 2
            assert report["nodes_budget"] == 20000 and report["repeats"] == 3
            assert report["pv_threads"] == pv and report["batch"] == 128
            assert report["positions_sha256"] == file_sha256(FIXTURES)
            assert report["new_sha256"] == expected["sha256"]
            assert report["model_sha256"] == expected["model_sha256"]
            assert report["old_sha256"] == file_sha256(report["old"])
            counts = Counter(r["index"] for r in report["records"])
            assert len(counts) == 30 and set(counts.values()) == {3}
            assert len({(r["index"], r["repeat"]) for r in report["records"]}) == 90
            measurements.append({
                "model": model, "pv_threads": pv, "report": str(path.relative_to(ROOT)),
                "old_sha256": report["old_sha256"], "new_sha256": report["new_sha256"],
                "samples": 90, "full_budget_samples": report["full_budget_positions"],
                "early_solved_samples": 90 - report["full_budget_positions"],
                "median_old_nps": report["median_old_nps"],
                "median_new_nps": report["median_new_nps"],
                "ratio_of_nps_medians": report["median_new_nps"] / report["median_old_nps"],
                "median_paired_ratio": report["median_position_ratio"],
                "per_repeat_median_ratio": report["per_repeat_median_ratio"],
            })
    unit = (args.reports / "unit.log").read_bytes()
    unit_text = unit.decode("utf-16" if unit.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8")
    assert "classic adapter tests passed" in unit_text
    auxiliary = {}
    for model in ("ny", "no_ny"):
        book = load(args.reports / f"book_{model}.json")
        assert book["entries"] > 0 and book["readback"] and book["policy_blend"]
        assert book["makebook_policy_blend"]
        reader = verification["ny" if model == "ny" else "after_no_ny"]
        assert book["reader_sha256"] == reader["sha256"]
        assert book["engine_sha256"] == file_sha256(
            args.reports / f"makebook_{model}/dlshogi_tensorrt.exe")
        auxiliary[f"book_{model}"] = book
    multi = load(args.reports / "multiponder.json")
    assert multi["hits"] == multi["misses"] == 10
    assert multi["quit_during_ponder"] and multi["children_exited"]
    assert multi["auxiliary_sha256"] == verification["ny"]["sha256"]
    assert multi["engine_sha256"] == file_sha256(
        args.reports / "multiponder_ny/usi_multiponder.exe")
    auxiliary["multiponder"] = {k: v for k, v in multi.items() if k != "records"}
    for variant, executable in (("tensorrt_double", "dlshogi_tensorrt_double.exe"),
                                ("tensorrt_debug", "dlshogi_tensorrt.exe"),
                                ("multiponder_no_ny", "usi_multiponder.exe")):
        log = (args.reports / f"smoke_{variant}.log").read_bytes()
        text = log.decode("utf-16" if log.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        assert "bestmove " in text
        auxiliary[variant] = {"sha256": file_sha256(args.reports / variant / executable),
                              "smoke_threads": 2}
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True).strip()
    result = {
        "complete": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_and_driver": gpu,
        "backend": "TensorRT 8.6.1.6", "threads": 2, "nodes": 20000,
        "batch": 128, "positions": 30, "repeats": 3,
        "fixture_sha256": file_sha256(FIXTURES),
        "verification": verification, "auxiliary": auxiliary, "measurements": measurements,
        "notes": [
            "Sequential old/new GPU runs; no concurrent builds or other engine tests.",
            "One engine process per side is retained for all repeats: subtree reuse is disabled, NN caches persist, and initial model loading/warmup is excluded.",
            "Early-solved samples are recorded but excluded from NPS aggregates.",
            "Discarded interrupted preliminary run overlapped compilation and is not included.",
            "These runs compare legacy search with the corrected classic integration; they do not isolate the speedup of each fix.",
            "IdlingMinimumWork=0 and MaxPrefetch=32 remain unchanged; no Elo or speed gate.",
            "Earlier 40-game evidence in classic_acceptance.json concerns earlier executable hashes.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
