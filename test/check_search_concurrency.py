"""Functional/stats matrix for classic helpers and optional real multi-GPU runs.

Use >=30 positions, >=20000 visits and >=3 repeats for performance assessment.
The default small run is a concurrency regression, not a speed acceptance gate.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import cshogi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, FIXTURES, file_sha256, validate_move


def measure(args, executable, slots, helpers, searchers, legacy=False):
    options = {"DNN_Cache_Size": args.cache_size}
    for gpu in range(8):
        suffix = str(gpu + 1) if gpu else ""
        options[("UCT_Threads" if legacy else "DNN_Slots") + suffix] = slots if gpu in args.gpus else 0
        options["DNN_Batch_Size" + suffix] = args.batch
    if not legacy:
        options.update(Threads=args.threads, MinibatchSize=args.minibatch_size,
                       TaskWorkers=helpers, MaxConcurrentSearchers=searchers)
    engine = Engine(executable, args.model, threads=slots, batch=args.batch,
                    pv_threads=args.pv_threads, extra_options=options)
    records = []
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"]
    try:
        if not legacy:
            assert not any(name.startswith("UCT_Threads") for name in engine.option_names)
            for name in ("Threads", "DNN_Slots", "MinibatchSize", "TaskWorkers", "MaxConcurrentSearchers"):
                assert any(line.startswith("option name " + name + " ")
                           for line in engine.transcript), "missing option " + name
        previous = engine.nn_stats()
        for repeat in range(args.repeats):
            for ordinal in range(args.positions):
                index = ordinal * len(fixtures) // args.positions
                fixture = fixtures[index]
                started = time.perf_counter()
                best, lines = engine.go(fixture["moves"], f"nodes {args.nodes}", timeout=180)
                response_seconds = time.perf_counter() - started
                current = engine.nn_stats()  # Drain all workers before reading.
                drained_seconds = time.perf_counter() - started
                board = cshogi.Board(fixture["sfen"])
                validate_move(board, best)
                for line in lines:
                    if " pv " in line:
                        pv_board = board.copy()
                        for move in line.split(" pv ", 1)[1].split():
                            pv_board.push(validate_move(pv_board, move))
                visits = [int(line.split(" nodes ", 1)[1].split()[0])
                          for line in lines if " nodes " in line]
                evaluations = current["evaluations"] - previous["evaluations"]
                batches = current["batches"] - previous["batches"]
                assert visits and visits[-1] > 0 and evaluations >= 0 and batches >= 0
                assert (evaluations == 0) == (batches == 0)
                assert current == engine.nn_stats(), "NN work still running after drain"
                records.append(dict(index=index, repeat=repeat, bestmove=best,
                                    nodes=visits[-1], evaluations=evaluations, batches=batches,
                                    response_seconds=response_seconds, drained_seconds=drained_seconds))
                previous = current
    finally:
        engine.close()
    seconds = sum(r["drained_seconds"] for r in records)
    evaluations = sum(r["evaluations"] for r in records)
    batches = sum(r["batches"] for r in records)
    return dict(engine_sha256=file_sha256(executable), slots_per_gpu=slots,
                threads=slots * len(args.gpus) if legacy else args.threads,
                task_workers=None if legacy else helpers,
                max_concurrent_searchers=None if legacy else searchers,
                evaluations_per_second=evaluations / seconds,
                average_batch=evaluations / batches if batches else 0,
                nodes_per_second=sum(r["nodes"] for r in records) / seconds,
                response_seconds=sum(r["response_seconds"] for r in records),
                drained_seconds=seconds, records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--old", help="Optional legacy binary with the nnstats instrumentation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU IDs, e.g. 0,1")
    parser.add_argument("--positions", type=int, default=6)
    parser.add_argument("--nodes", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0, help="Global search workers; 0 uses upstream auto")
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=100000)
    parser.add_argument("--pv-threads", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.gpus = sorted({int(gpu) for gpu in args.gpus.split(",")})
    assert args.gpus and all(0 <= gpu < 8 for gpu in args.gpus)
    assert 1 <= args.positions <= 100 and args.nodes >= 1 and args.repeats >= 1
    args.report.unlink(missing_ok=True)
    results = []
    for slots, helpers, searchers in itertools.product((1, 2), (0, -1), (1, 0)):
        results.append(measure(args, args.engine, slots, helpers, searchers))
        print(f"passed slots/GPU={slots} Threads={args.threads} helpers={helpers} searchers={searchers}", flush=True)
    legacy = []
    if args.old:
        for threads in (1, 2):
            legacy.append(measure(args, args.old, threads, 0, 1, legacy=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(
        complete=True, gpu_ids=args.gpus, physical_gpu_count=len(args.gpus),
        model_sha256=file_sha256(args.model), fixture_sha256=file_sha256(FIXTURES),
        positions=args.positions, nodes_budget=args.nodes, repeats=args.repeats,
        batch=args.batch, threads=args.threads, minibatch_size=args.minibatch_size,
        cache_size=args.cache_size, pv_threads=args.pv_threads,
        timing="NN and node rates use wall time through worker drain; response time is recorded separately.",
        results=results, legacy=legacy), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
