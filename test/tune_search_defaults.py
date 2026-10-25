"""Sequential single-GPU default tuning; raw fixed-position data is retained."""
import argparse
import json
from pathlib import Path
import sys
import time
import cshogi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, FIXTURES, file_sha256, validate_move

CASES = {
    "legacy": dict(threads=2, slots=2, helpers=0),
    "reference": dict(threads=2, slots=2, helpers=0),
    "auto_s1_t0": dict(threads=0, slots=1, helpers=0),
    "auto_s1_auto": dict(threads=0, slots=1, helpers=-1),
    "auto_s2_t0": dict(threads=0, slots=2, helpers=0),
    "auto_s2_auto": dict(threads=0, slots=2, helpers=-1),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--old", required=True, help="Legacy binary with completed-forward nnstats")
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate", choices=list(CASES)[2:], help="Confirm this screened automatic setting")
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.positions <= 100 and args.repeats >= 1
    names = ["legacy", "reference", args.candidate] if args.candidate else list(CASES)
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"]
    engines, previous, records = {}, {}, []
    report = dict(complete=False, phase="confirmation" if args.candidate else "screening",
                  gpu_count=1, model_sha256=file_sha256(args.model), fixture_sha256=file_sha256(FIXTURES),
                  nodes_budget=args.nodes, positions=args.positions, repeats=args.repeats,
                  engine_sha256=file_sha256(args.engine), legacy_sha256=file_sha256(args.old),
                  settings={name:CASES[name] for name in names}, records=records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in names:
            config = CASES[name]
            engine = Engine(args.old if name == "legacy" else args.engine, args.model,
                            threads=2 if name == "legacy" else config["threads"], slots=config["slots"],
                            extra_options={} if name == "legacy" else {
                                "TaskWorkers":config["helpers"], "MinibatchSize":0, "MaxConcurrentSearchers":1})
            engines[name] = engine
            engine.go(fixtures[-1]["moves"], "nodes 1000", timeout=120)
            previous[name] = engine.nn_stats()
        for repeat in range(args.repeats):
            for ordinal in range(args.positions):
                index = ordinal * len(fixtures) // args.positions
                fixture = fixtures[index]
                row = dict(index=index, repeat=repeat)
                offset = (ordinal + repeat) % len(names)
                for name in names[offset:] + names[:offset]:
                    engine = engines[name]
                    started = time.perf_counter()
                    best, lines = engine.go(fixture["moves"], f"nodes {args.nodes}", timeout=180)
                    response = time.perf_counter() - started
                    current = engine.nn_stats()
                    seconds = time.perf_counter() - started
                    validate_move(cshogi.Board(fixture["sfen"]), best)
                    visits = [int(line.split(" nodes ",1)[1].split()[0]) for line in lines if " nodes " in line]
                    row[name] = dict(bestmove=best, nodes=visits[-1], seconds=seconds, response_seconds=response,
                                     evaluations=current["evaluations"]-previous[name]["evaluations"],
                                     batches=current["batches"]-previous[name]["batches"])
                    previous[name] = current
                records.append(row)
                args.report.write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
                print(f"{repeat+1}/{args.repeats} position {ordinal+1}/{args.positions}",flush=True)
        measured = [row for row in records if all(row[name]["nodes"] >= args.nodes for name in names)]
        assert measured
        report["full_budget_samples"] = len(measured)
        report["totals"] = {}
        for name in names:
            totals = {key:sum(row[name][key] for row in measured)
                      for key in ("nodes","seconds","response_seconds","evaluations","batches")}
            totals.update(nps=totals["nodes"]/totals["seconds"],
                          nn_per_second=totals["evaluations"]/totals["seconds"],
                          average_batch=totals["evaluations"]/totals["batches"])
            report["totals"][name] = totals
        report["complete"] = True
        args.report.write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report["totals"],indent=2),flush=True)
    finally:
        for engine in engines.values():
            engine.close()

if __name__ == "__main__":
    main()
