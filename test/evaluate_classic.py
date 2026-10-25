"""Independent USI legality, fixed-position regression and paired-match runner.

Requires cshogi. Reports JSON evidence under artifacts; no Elo inference.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import statistics
import subprocess
import threading
import time

import cshogi

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "test" / "classic_positions.json"


def file_sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Engine:
    def __init__(self, executable, model, threads=1, batch=128, mate=0, pv_threads=0, extra_options=None, slots=None):
        env = os.environ.copy()
        env["PATH"] = r"C:\TensorRT-8.6.1.6\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;" + env.get("PATH", "")
        self.process = subprocess.Popen([str(Path(executable).resolve())], cwd=ROOT, env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.lines = queue.Queue()
        self.errors = []
        self.transcript = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self.errors.extend(self.process.stderr), daemon=True).start()
        try:
            self.send("usi")
            self.until(lambda line: line == "usiok", 15)
            self.option_names = {line.split(" type ", 1)[0][len("option name "):]
                                 for line in self.transcript if line.startswith("option name ")}
            options = {"DNN_Model": str(Path(model).resolve()), "DNN_Batch_Size": batch,
                       "Mate_Root_Search": mate, "DfPn_Hash": 64, "PV_Mate_Search_Threads": pv_threads,
                       "Resign_Threshold": 0, "OwnBook": "false", "Use_Book_Policy": "false",
                       "Random_Ply": 0, "Random2_Ply": 0, "USI_Ponder": "false", "Stochastic_Ponder": "false",
                       "Draw_Ply": 256, "Time_Margin": 0, "Byoyomi_Margin": 0, "ReuseSubtree": "false"}
            if "DNN_Slots" in self.option_names:
                # Fixed-position comparisons pin helpers off; tuning/default
                # tests explicitly override these independent settings.
                options.update(Threads=threads, DNN_Slots=slots if slots is not None else 1,
                               TaskWorkers=0)
            else:
                # Archived legacy binaries retain GPU-bound UCT threads.
                options["UCT_Threads"] = threads
            options.update(extra_options or {})
            for name, value in options.items():
                self.send(f"setoption name {name} value {value}")
            self.send("isready")
            self.until(lambda line: line == "readyok", 1800)
            self.send("usinewgame")
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def send(self, text):
        self.transcript.append("> " + text)
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        result = []
        while True:
            try:
                line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as error:
                raise TimeoutError("USI timeout: " + "\n".join(result[-5:])) from error
            if line is None:
                raise RuntimeError(f"Engine exited {self.process.poll()}: {''.join(self.errors)}")
            self.transcript.append(line)
            if " wdl " in line:
                raise AssertionError("Unexpected USI WDL output: " + line)
            if " nps " in line:
                assert int(line.split(" nps ", 1)[1].split()[0]) >= 0, "Unavailable NPS must be omitted"
            result.append(line)
            if predicate(line):
                return result

    def go(self, moves, command="nodes 800", timeout=30):
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self.send("go " + command)
        lines = self.until(lambda line: line.startswith("bestmove "), timeout)
        return lines[-1].split()[1], lines

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
                raise AssertionError("Engine failed to exit cleanly on quit")
        self.process.stdin.close()
        assert self.process.returncode == 0, f"Engine exit code {self.process.returncode}: {''.join(self.errors)}"

    def nn_stats(self):
        self.send("nnstats")
        line = self.until(lambda line: line.startswith("info string nnstats evaluations "), 30)[-1]
        fields = line.split()
        assert fields[5] == "batches"
        return {"evaluations": int(fields[4]), "batches": int(fields[6])}


def generate():
    rng = random.Random(20261024)
    groups = [[] for _ in range(4)]
    seen = set()
    ranges = [(16, 40), (41, 80), (81, 120), (121, 180)]
    for game in range(10000):
        board = cshogi.Board()
        moves = []
        sample_plies = [rng.randint(low, high) for low, high in ranges]
        for ply in range(1, 181):
            legal = list(board.legal_moves)
            if not legal or board.is_draw() != cshogi.NOT_REPETITION:
                break
            move = rng.choice(legal)
            moves.append(cshogi.move_to_usi(move))
            board.push(move)
            if board.is_check() or len(list(board.legal_moves)) < 10 or board.is_game_over() or board.mate_move_in_1ply():
                continue
            key = board.zobrist_hash()
            for index, (low, high) in enumerate(ranges):
                if ply == sample_plies[index] and len(groups[index]) < 25 and key not in seen:
                    groups[index].append({"moves": list(moves), "sfen": board.sfen(), "stage": index})
                    seen.add(key)
                    break
        if all(len(group) == 25 for group in groups):
            break
    assert all(len(group) == 25 for group in groups)
    fixture = {"seed": 20261024, "positions": [position for group in groups for position in group]}
    FIXTURES.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"fixed {len(fixture['positions'])} positions: {FIXTURES}", flush=True)


def validate_move(board, text):
    if text == "resign":
        raise AssertionError("Unexpected resignation with threshold disabled")
    if text == "win":
        assert board.is_nyugyoku(), "Unjustified declaration"
        return None
    move = board.move_from_usi(text)
    assert move in list(board.legal_moves), f"Illegal move {text} in {board.sfen()}"
    return move


def compare(args):
    positions = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"]
    old = Engine(args.old, args.model, batch=args.batch)
    new = None
    records = []
    try:
        new = Engine(args.new, args.model, batch=args.batch)
        for index, position in enumerate(positions[:args.positions]):
            board = cshogi.Board(position["sfen"])
            old_move, _ = old.go(position["moves"], f"nodes {args.nodes}")
            new_move, _ = new.go(position["moves"], f"nodes {args.nodes}")
            validate_move(board, old_move)
            validate_move(board, new_move)
            records.append({"index": index, "old": old_move, "new": new_move, "match": old_move == new_move})
            print(f"position {index + 1}/{args.positions}: {old_move} / {new_move}", flush=True)
        rate = sum(record["match"] for record in records) / len(records)
        return {"kind": "fixed_nodes", "nodes": args.nodes, "threads": 1, "mate": 0,
                "positions_sha256": hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
                "match_rate": rate, "requires_investigation": rate < .5, "records": records}
    finally:
        old.close()
        if new:
            new.close()


def nncompare(args):
    positions = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"]
    old = Engine(args.old, args.model, batch=args.batch)
    new = None
    records = []
    def parse(line):
        fields = line.split()
        return float(fields[4]), {move: float(value) for move, value in (field.split(":") for field in fields[5:])}
    try:
        # A cached root may be backed up out of order together with descendants
        # before classic reaches its next stopper check. Disable the one dlshogi
        # cache here so the one-visit test has exactly one root evaluation.
        new = Engine(args.new, args.model, batch=args.batch,
                     extra_options={"DNN_Cache_Size": 0, "MultiPV": 3})
        for index, position in enumerate(positions[:args.positions]):
            _, lines = old.go(position["moves"], "nodes 1")
            probes = [line for line in lines if line.startswith("info string nnprobe value ")]
            assert len(probes) == 1, "Use the instrumented pinned baseline from build_legacy_probe.py"
            new.send("position startpos moves " + " ".join(position["moves"]))
            new.send("nnprobe")
            actual = new.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1]
            old_value, old_policy = parse(probes[0])
            new_value, new_policy = parse(actual)
            assert old_policy.keys() == new_policy.keys(), "NN policy move mapping mismatch"
            value_error = abs(old_value - new_value)
            policy_error = max(abs(old_policy[move] - new_policy[move]) for move in old_policy)
            assert value_error <= 1e-3 and policy_error <= 1e-3, (index, value_error, policy_error)
            # Policy ties within the inference tolerance are interchangeable.
            top = max(new_policy, key=new_policy.get)
            assert max(old_policy.values()) - old_policy[top] <= 1e-3
            best, search_lines = new.go(position["moves"], "nodes 1")
            assert best in old_policy and max(old_policy.values()) - old_policy[best] <= 1e-3, "One-visit root policy order mismatch"
            expected_priors = sorted(old_policy.values(), reverse=True)
            top_moves = []
            for rank in (1, 2, 3):
                lines = [line for line in search_lines if f" multipv {rank} " in line and " pv " in line]
                assert lines, (index, "missing one-visit root policy rank", rank)
                move = lines[-1].split(" pv ", 1)[1].split()[0]
                assert move in old_policy and abs(old_policy[move] - expected_priors[rank - 1]) <= 1e-3
                top_moves.append(move)
            assert len(set(top_moves)) == 3, "Duplicate root policy entries"
            scores = [int(line.split(" score cp ", 1)[1].split()[0]) for line in search_lines
                      if " score cp " in line and (" multipv " not in line or " multipv 1 " in line)]
            expected_cp = -30000 if new_value <= 0 else 30000 if new_value >= 1 else int(756 * math.log(new_value / (1 - new_value)))
            visits = [int(line.split(" nodes ", 1)[1].split()[0]) for line in search_lines if " nodes " in line]
            assert visits and visits[-1] == 1, (index, "expected exactly one root visit", visits)
            assert scores and abs(scores[-1] - expected_cp) <= 1, (index, "one-visit root scalar sign", scores, expected_cp)
            records.append({"index": index, "value_error": value_error, "policy_error": policy_error,
                            "one_visit_move": best, "one_visit_top3": top_moves,
                            "one_visit_cp_error": abs(scores[-1] - expected_cp)})
            print(f"NN position {index+1}: value error={value_error:.3g}, policy error={policy_error:.3g}", flush=True)
        return {"kind": "legacy_root_nn", "requires_investigation": False, "records": records}
    finally:
        old.close()
        if new:
            new.close()


def match(args):
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))
    openings = [position for position in fixture["positions"] if position["stage"] == 0][:20]
    old = Engine(args.old, args.model, mate=33, batch=args.batch)
    new = None
    records = []
    try:
        new = Engine(args.new, args.model, mate=33, batch=args.batch)
        for game in range(args.games):
            moves = list(openings[(game // 2) % len(openings)]["moves"])
            board = cshogi.Board()
            occurrences = Counter({board.zobrist_hash(): 1})
            for text in moves:
                board.push_usi(text)
                occurrences[board.zobrist_hash()] += 1
            new_color = game % 2
            engines = [old, new] if new_color else [new, old]
            for engine in engines:
                engine.send("usinewgame")
            score = .5
            reason = "move_limit"
            while board.move_number <= 256:
                color = board.turn
                legal = list(board.legal_moves)
                if not legal:
                    score = 0.0 if color == new_color else 1.0
                    reason = "mate"
                    break
                repetition = board.is_draw()
                # cshogi's search repetition test returns on the first cycle.
                # Actual game adjudication requires four occurrences.
                if repetition in (cshogi.REPETITION_DRAW, cshogi.REPETITION_WIN, cshogi.REPETITION_LOSE) and occurrences[board.zobrist_hash()] >= 4:
                    if repetition == cshogi.REPETITION_WIN:
                        score = 1.0 if color == new_color else 0.0
                    elif repetition == cshogi.REPETITION_LOSE:
                        score = 0.0 if color == new_color else 1.0
                    reason = "repetition"
                    break
                text, _ = engines[color].go(moves, f"btime 0 wtime 0 byoyomi {args.byoyomi}", timeout=5)
                move = validate_move(board, text)
                if move is None:
                    score = 1.0 if color == new_color else 0.0
                    reason = "declaration"
                    break
                board.push(move)
                occurrences[board.zobrist_hash()] += 1
                moves.append(text)
            records.append({"game": game, "new_color": new_color, "score": score, "reason": reason, "moves": moves})
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({"complete": False, "kind": "paired_smoke_match", "records": records}, indent=2) + "\n", encoding="utf-8")
            print(f"game {game + 1}/{args.games}: new={score} {reason}", flush=True)
        rate = sum(record["score"] for record in records) / len(records)
        return {"kind": "completion_match" if args.completion_only else "paired_smoke_match",
                "games": args.games, "byoyomi_ms": args.byoyomi,
                "score_rate": rate, "requires_investigation": not args.completion_only and rate < .2, "records": records}
    finally:
        old.close()
        if new:
            new.close()


def benchmark(args):
    """Run only while no other GPU jobs are active; this has no speed gate."""
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"]
    assert 1 <= args.positions <= len(fixture) and args.repeats >= 1
    assert args.threads >= 1 and args.pv_threads >= 0
    old = Engine(args.old, args.model, batch=args.batch, threads=args.threads, pv_threads=args.pv_threads)
    new = None
    records = []
    try:
        settings = {"Threads": args.new_threads if args.new_threads is not None else args.threads,
                    "MinibatchSize": args.minibatch_size, "TaskWorkers": args.task_workers}
        new = Engine(args.new, args.model, batch=args.batch, threads=args.threads,
                     pv_threads=args.pv_threads, slots=args.slots, extra_options=settings)
        for engine in (old, new):
            engine.go(fixture[-1]["moves"], "nodes 1000")
        previous_stats = ({label: engine.nn_stats() for label, engine in (("old", old), ("new", new))}
                          if args.nn_stats else {})
        for sample in range(args.positions * args.repeats):
            repeat, ordinal = divmod(sample, args.positions)
            index = ordinal * len(fixture) // args.positions
            position = fixture[index]
            record = {"index": index, "repeat": repeat}
            # Alternate execution order to reduce warmup/temperature bias.
            sequence = [("old", old), ("new", new)]
            if (ordinal + repeat) % 2:
                sequence.reverse()
            for label, engine in sequence:
                begin = time.perf_counter()
                move, lines = engine.go(position["moves"], f"nodes {args.nodes}")
                elapsed = time.perf_counter() - begin
                validate_move(cshogi.Board(position["sfen"]), move)
                nodes = [int(line.split(" nodes ", 1)[1].split()[0]) for line in lines if " nodes " in line]
                assert nodes and nodes[-1] > 0
                record[label] = {"nodes": nodes[-1], "seconds": elapsed,
                                 "wall_nps": nodes[-1] / elapsed, "bestmove": move}
                if args.nn_stats:
                    current = engine.nn_stats()
                    previous = previous_stats[label]
                    evaluations = current["evaluations"] - previous["evaluations"]
                    batches = current["batches"] - previous["batches"]
                    assert evaluations >= batches >= 0
                    assert (evaluations == 0) == (batches == 0)
                    previous_stats[label] = current
                    record[label].update(nn_evaluations=evaluations, nn_batches=batches,
                                         nn_evaluations_per_second=evaluations / elapsed,
                                         mean_batch_size=evaluations / batches if batches else None,
                                         nodes_per_nn_evaluation=nodes[-1] / evaluations if evaluations else None)
            records.append(record)
            print(f"benchmark {len(records)}/{args.positions * args.repeats}: old={record['old']['wall_nps']:.0f} new={record['new']['wall_nps']:.0f}", flush=True)
            if args.nn_stats:
                print(f"  NN/s old={record['old']['nn_evaluations_per_second']:.0f} "
                      f"new={record['new']['nn_evaluations_per_second']:.0f}; "
                      f"mean batch old={record['old']['mean_batch_size']} "
                      f"new={record['new']['mean_batch_size']}", flush=True)
        # Solved positions can stop early in classic and then are not comparable
        # to the old engine filling its visit budget with terminal revisits.
        measured = [record for record in records if all(record[label]["nodes"] >= args.nodes for label in ("old", "new"))]
        assert measured, "All benchmark positions solved early; use a broader position sample"
        inference = None
        if args.nn_stats:
            inference = {}
            for label in ("old", "new"):
                evaluations = sum(record[label]["nn_evaluations"] for record in measured)
                batches = sum(record[label]["nn_batches"] for record in measured)
                seconds = sum(record[label]["seconds"] for record in measured)
                nodes = sum(record[label]["nodes"] for record in measured)
                assert evaluations > 0 and batches > 0
                inference[label] = {"evaluations": evaluations, "batches": batches,
                                    "seconds": seconds, "nodes": nodes,
                                    "evaluations_per_second": evaluations / seconds,
                                    "aggregate_wall_nps": nodes / seconds,
                                    "mean_batch_size": evaluations / batches,
                                    "nodes_per_evaluation": nodes / evaluations}
            positive = [record for record in measured if all(record[label]["nn_evaluations"] > 0 for label in ("old", "new"))]
            inference["paired_positive_samples"] = len(positive)
            inference["median_paired_rate_ratio"] = statistics.median(
                record["new"]["nn_evaluations_per_second"] / record["old"]["nn_evaluations_per_second"]
                for record in positive) if positive else None
            inference["aggregate_rate_ratio"] = inference["new"]["evaluations_per_second"] / inference["old"]["evaluations_per_second"]
            inference["visit_count_factor"] = inference["new"]["nodes_per_evaluation"] / inference["old"]["nodes_per_evaluation"]
            inference["aggregate_nps_ratio"] = inference["new"]["aggregate_wall_nps"] / inference["old"]["aggregate_wall_nps"]
            inference["per_repeat_rate_ratio"] = []
            for repeat in range(args.repeats):
                subset = [record for record in measured if record["repeat"] == repeat]
                if not subset:
                    inference["per_repeat_rate_ratio"].append(None)
                    continue
                rates = {label: sum(record[label]["nn_evaluations"] for record in subset) /
                         sum(record[label]["seconds"] for record in subset) for label in ("old", "new")}
                inference["per_repeat_rate_ratio"].append(rates["new"] / rates["old"] if rates["old"] else None)
        return {"kind": "sequential_wall_nps", "nodes_budget": args.nodes,
                "inference": inference,
                "threads": args.threads, "new_threads": args.new_threads if args.new_threads is not None else args.threads,
                "new_slots": args.slots, "task_workers": args.task_workers, "minibatch_size": args.minibatch_size,
                "pv_threads": args.pv_threads, "repeats": args.repeats, "mate": 0, "positions_sha256": file_sha256(FIXTURES),
                "full_budget_positions": len(measured),
                "unique_full_budget_positions": len({record["index"] for record in measured}),
                "per_repeat_median_ratio": [statistics.median(
                    record["new"]["wall_nps"] / record["old"]["wall_nps"]
                    for record in measured if record["repeat"] == repeat)
                    if any(record["repeat"] == repeat for record in measured) else None
                    for repeat in range(args.repeats)],
                "early_solved_positions": [record["index"] for record in records if record not in measured],
                "median_old_nps": statistics.median(record["old"]["wall_nps"] for record in measured),
                "median_new_nps": statistics.median(record["new"]["wall_nps"] for record in measured),
                "median_position_ratio": statistics.median(record["new"]["wall_nps"] / record["old"]["wall_nps"] for record in measured),
                "requires_investigation": False, "speed_acceptance_threshold": None, "records": records}
    finally:
        old.close()
        if new:
            new.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "compare", "match", "nncompare", "benchmark"])
    parser.add_argument("--old")
    parser.add_argument("--new")
    parser.add_argument("--model")
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--new-threads", type=int, help="Benchmark new engine global workers; 0 = auto")
    parser.add_argument("--slots", type=int, default=2, help="New engine inference slots per GPU (benchmark)")
    parser.add_argument("--task-workers", type=int, default=0, help="New engine helpers (benchmark)")
    parser.add_argument("--minibatch-size", type=int, default=0, help="New engine collection batch (benchmark)")
    parser.add_argument("--pv-threads", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nn-stats", action="store_true", help="Benchmark completed NN evaluations and batches; both binaries must support nnstats")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--byoyomi", type=int, default=1000)
    parser.add_argument("--completion-only", action="store_true", help="Same-binary backend completion test; no strength threshold")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "comparison.json")
    args = parser.parse_args()
    if args.nn_stats:
        assert args.mode == "benchmark"
    if args.completion_only:
        assert args.mode == "match" and Path(args.old).resolve() == Path(args.new).resolve()
    if args.mode == "generate":
        generate()
        return
    args.report.unlink(missing_ok=True)
    result = {"compare": compare, "match": match, "nncompare": nncompare, "benchmark": benchmark}[args.mode](args)
    result.update(complete=True, batch=args.batch, old=str(Path(args.old).resolve()), new=str(Path(args.new).resolve()), model=str(Path(args.model).resolve()),
                  old_sha256=file_sha256(args.old), new_sha256=file_sha256(args.new), model_sha256=file_sha256(args.model))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)
    if result["requires_investigation"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
