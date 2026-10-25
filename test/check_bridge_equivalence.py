"""Compare bridge-only optimizations against a frozen classic executable."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, FIXTURES, file_sha256, validate_move
import cshogi


def principal_variation(lines):
    candidates = [line for line in lines if " pv " in line and
                  (" multipv " not in line or " multipv 1 " in line)]
    assert candidates
    return candidates[-1].split(" pv ", 1)[1].split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--positions", type=int, default=30)
    parser.add_argument("--nodes", type=int, default=4000)
    parser.add_argument("--require-exact-nn-counts", action="store_true",
                        help="Also require count equality; stop-time prefetch can vary with real GPU scheduling")
    parser.add_argument("--observe-only", action="store_true", help="Record baseline variability without asserting output equality")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"]
    assert 1 <= args.positions <= len(fixture)
    old = Engine(args.old, args.model)
    new = None
    records = []
    try:
        new = Engine(args.new, args.model)
        previous = {"old": old.nn_stats(), "new": new.nn_stats()}
        for ordinal in range(args.positions):
            index = ordinal * len(fixture) // args.positions
            position = fixture[index]
            result = {"index": index}
            for label, engine in (("old", old), ("new", new)):
                best, lines = engine.go(position["moves"], f"nodes {args.nodes}", timeout=120)
                validate_move(cshogi.Board(position["sfen"]), best)
                current = engine.nn_stats()
                visits = [int(line.split(" nodes ", 1)[1].split()[0]) for line in lines if " nodes " in line]
                result[label] = {"bestmove": best, "pv": principal_variation(lines),
                                 "nodes": visits[-1],
                                 "nn_evaluations": current["evaluations"] - previous[label]["evaluations"],
                                 "nn_batches": current["batches"] - previous[label]["batches"]}
                previous[label] = current
            # CPU-only terminal/cache-hit backups can overshoot a node budget
            # differently while the watchdog observes it. Compare the actual
            # requested invariants, retaining visit counts for transparency.
            keys = ["bestmove", "pv"]
            if args.require_exact_nn_counts:
                keys += ["nn_evaluations", "nn_batches"]
            if not args.observe_only:
                for key in keys:
                    assert result["old"][key] == result["new"][key], (key, result)
            records.append(result)
            print(f"bridge equivalence {ordinal + 1}/{args.positions}", flush=True)
    finally:
        old.close()
        if new:
            new.close()
    # Exercise the cache's canonical move order through two different histories
    # reaching exactly the same board, rather than just repeated identical input.
    engine = Engine(args.new, args.model)
    try:
        probes, counters = [], []
        for moves in ("7g7f 3c3d 2g2f 8c8d", "2g2f 8c8d 7g7f 3c3d"):
            engine.send("position startpos moves " + moves)
            engine.send("nnprobe")
            probes.append(engine.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1])
            counters.append(engine.nn_stats())
        assert probes[0] == probes[1], "transposition changed cached policy/value order"
        assert counters[0] == counters[1], "transposed board was not served by the cache"
    finally:
        engine.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"complete": True, "threads": 1, "nodes_budget": args.nodes,
                                      "old_sha256": file_sha256(args.old), "new_sha256": file_sha256(args.new),
                                      "model_sha256": file_sha256(args.model), "fixture_sha256": file_sha256(FIXTURES),
                                      "visit_count_difference_samples": sum(r["old"]["nodes"] != r["new"]["nodes"] for r in records),
                                      "nn_count_difference_samples": sum(r["old"]["nn_evaluations"] != r["new"]["nn_evaluations"] for r in records),
                                      "nn_batch_difference_samples": sum(r["old"]["nn_batches"] != r["new"]["nn_batches"] for r in records),
                                      "bestmove_difference_samples": sum(r["old"]["bestmove"] != r["new"]["bestmove"] for r in records),
                                      "pv_difference_samples": sum(r["old"]["pv"] != r["new"]["pv"] for r in records),
                                      "exact_nn_counts_required": args.require_exact_nn_counts,
                                      "observe_only": args.observe_only,
                                      "transposed_cache_policy_equal": True, "records": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
