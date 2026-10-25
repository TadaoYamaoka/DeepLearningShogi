"""Collect completed acceptance runs without silently relaxing their gates."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import ROOT, FIXTURES, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=ROOT / "artifacts/verified")
    parser.add_argument("--matches", type=Path, default=ROOT / "artifacts/final")
    parser.add_argument("--output", type=Path, default=ROOT / "test/classic_acceptance.json")
    args = parser.parse_args()

    def read_log(name):
        raw = (args.reports / name).read_bytes()
        return raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8-sig")

    def read(folder, name, complete=False):
        data = json.loads((folder / name).read_text(encoding="utf-8"))
        if complete:
            assert data.get("complete") is True and data.get("requires_investigation") is False, name
        return data

    evidence = {}
    for variant in ("tensorrt_ny", "tensorrt_no_ny", "onnx_ny", "onnx_no_ny"):
        nn = read(args.reports, f"nn_{variant}.json", True)
        protocol = read(args.reports, f"protocol_{variant}.json")
        completion = read(args.reports, f"completion_{variant}.json", True)
        assert len(nn["records"]) == 100 and all(len(set(row["one_visit_top3"])) == 3 for row in nn["records"])
        assert all(row["value_error"] <= 1e-3 and row["policy_error"] <= 1e-3 and row["one_visit_cp_error"] <= 1 for row in nn["records"])
        assert len(protocol["passed"]) >= 21 and len(set(protocol["passed"])) == len(protocol["passed"])
        assert completion["games"] == len(completion["records"]) == 2
        assert nn["new_sha256"] == protocol["engine_sha256"] == completion["new_sha256"]
        assert nn["model_sha256"] == protocol["model_sha256"] == completion["model_sha256"]
        assert file_sha256(nn["new"]) == nn["new_sha256"], "Executable changed after verification"
        evidence[variant] = {
            "engine_sha256": nn["new_sha256"], "model_sha256": nn["model_sha256"],
            "legacy_nn_sha256": nn["old_sha256"], "nn_positions": len(nn["records"]),
            "max_value_error": max(row["value_error"] for row in nn["records"]),
            "max_policy_error": max(row["policy_error"] for row in nn["records"]),
            "max_one_visit_cp_error": max(row["one_visit_cp_error"] for row in nn["records"]),
            "protocol_checks": protocol["passed"], "completed_games": completion["games"]}

    for suffix in ("ny", "no_ny"):
        comparison = read(args.reports, f"search_{suffix}.json", True)
        match = read(args.matches, f"match_{suffix}.json", True)
        speed = read(args.reports, f"nps_{suffix}.json", True)
        assert len(comparison["records"]) == 100 and comparison["nodes"] == 800 and comparison["match_rate"] >= .5
        assert comparison["positions_sha256"] == file_sha256(FIXTURES)
        assert match["games"] == len(match["records"]) == 40 and match["byoyomi_ms"] == 1000 and match["score_rate"] >= .2
        assert speed["new_sha256"] == comparison["new_sha256"] == evidence[f"tensorrt_{suffix}"]["engine_sha256"]
        assert comparison["old_sha256"] == match["old_sha256"] == speed["old_sha256"]
        evidence[f"tensorrt_{suffix}"].update(
            bestmove_match_rate=comparison["match_rate"], paired_games=40,
            paired_score_rate=match["score_rate"], paired_match_engine_sha256=match["new_sha256"],
            median_old_wall_nps=speed["median_old_nps"], median_new_wall_nps=speed["median_new_nps"],
            median_position_nps_ratio=speed["median_position_ratio"])
        book = read(args.reports, f"makebook_{suffix}.json")
        assert book["entries"] > 0 and book["readback"] and book["policy_blend"] and book["makebook_policy_blend"]
        assert book["reader_sha256"] == comparison["new_sha256"]
        evidence[f"makebook_{suffix}"] = book

    multi = read(args.reports, "multiponder.json")
    assert multi["hits"] == multi["misses"] == 10 and multi["children_exited"] and multi["quit_during_ponder"]
    assert len(multi["records"]) == 20 and multi["auxiliary_sha256"] == evidence["tensorrt_ny"]["engine_sha256"]
    evidence["multiponder"] = {key: value for key, value in multi.items() if key != "records"}
    parallel = read(args.reports, "protocol_multithread.json")
    assert parallel["threads"] == 2 and len(parallel["passed"]) >= 21
    assert parallel["engine_sha256"] == evidence["tensorrt_ny"]["engine_sha256"]
    evidence["parallel_search"] = {"threads": 2, "protocol_checks": parallel["passed"]}
    for configuration in ("release", "debug"):
        log = read_log(f"classic_tests_{configuration}.log")
        assert "classic adapter tests passed; max fixture=562 node=64 edge=4" in log
    for variant in ("tensorrt_debug", "tensorrt_double", "multiponder_no_ny"):
        log = read_log(f"smoke_{variant}.runner.log")
        assert "USI smoke passed:" in log

    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "dlshogi_baseline": "0cc16854474ab3c7c4f7be66c624f9f9001f7543",
        "lc0_upstream": "58e780090b8ffb765743b73a794f994c3a8c9476",
        "fixture_sha256": file_sha256(FIXTURES),
        "gates": {"nn_tolerance": .001, "one_visit_top_k": 3, "one_visit_cp_error": 1,
                  "fixed_nodes_bestmove_match_minimum": .5, "paired_40_game_score_minimum": .2,
                  "speed_threshold": None, "elo_claim": False},
        "paired_match_note": "40-game runs use the frozen search executable in artifacts/final. Subsequent changes add USI display checks, PV-dfpn proof safeguards and locking for leaf-terminal publication. PV workers are disabled and scores are not used for adjudication in these matches. The final executable is separately checked by the 100-position tests, protocol tests and two completion games per backend/model.",
        "performance_note": "Wall NPS is measured sequentially after other GPU jobs finish. Paired matches may run concurrently and are catastrophic-regression gates, not Elo estimates.",
        "limitations": ["Linux Makefiles updated but not compiled on this Windows host.",
                        "Selfplay retains its separate MCTS; it was built but not migrated.",
                        "Binary distribution permissions for proprietary inference libraries require separate review."],
        "evidence": evidence}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All acceptance gates passed: {args.output}")


if __name__ == "__main__":
    main()
