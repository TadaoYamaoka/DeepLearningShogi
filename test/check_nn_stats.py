"""Real-backend checks for cumulative completed-inference counters."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, FIXTURES, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nnprobe", action="store_true", help="Also test classic's direct probe and cache-hit counters")
    parser.add_argument("--nodes", type=int, default=128, help="Visits per batch=1 search; reduce for slow backends")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert args.nodes >= 1
    args.report.unlink(missing_ok=True)
    moves = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"][0]["moves"]
    engine = Engine(args.engine, args.model, threads=2, slots=2, batch=1,
                    extra_options={"DNN_Cache_Size": 0})
    checks = []
    try:
        initial = engine.nn_stats()
        assert initial == engine.nn_stats(), "stats query performed inference"
        for iteration in range(2):
            engine.go(moves, f"nodes {args.nodes}")
            current = engine.nn_stats()
            evaluations = current["evaluations"] - initial["evaluations"]
            batches = current["batches"] - initial["batches"]
            assert evaluations == batches > 0, "batch=1 must have one evaluation per forward"
            assert current == engine.nn_stats(), "completed counters still changing after bestmove"
            initial = current
        checks.append("batch=1 evaluations equal batches across two searches")
        if args.nnprobe:
            engine.send("nnprobe")
            engine.until(lambda line: line.startswith("info string nnprobe value "), 30)
            current = engine.nn_stats()
            assert all(current[key] - initial[key] == 1 for key in ("evaluations", "batches"))
            checks.append("uncached single probe increments both counters by one")
    finally:
        engine.close()
    if args.nnprobe:
        engine = Engine(args.engine, args.model, threads=2, slots=2, batch=1)
        try:
            engine.send("position startpos moves " + " ".join(moves))
            previous = engine.nn_stats()
            # nnprobe needs the position command and leaves it in effect.
            snapshots = [previous]
            for _ in range(2):
                engine.send("nnprobe")
                engine.until(lambda line: line.startswith("info string nnprobe value "), 30)
                snapshots.append(engine.nn_stats())
            assert all(snapshots[1][key] - snapshots[0][key] == 1 for key in ("evaluations", "batches"))
            assert snapshots[1] == snapshots[2], "cache hit counted as inference"
            checks.append("cached probe does not increment counters")
        finally:
            engine.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"complete": True, "checks": checks, "nodes_per_search": args.nodes,
                                      "engine_sha256": file_sha256(args.engine),
                                      "model_sha256": file_sha256(args.model)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
