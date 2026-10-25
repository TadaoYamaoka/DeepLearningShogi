"""USI response gating, mate certificates, draw scoring and stop checks."""
import argparse
import json
import math
from pathlib import Path
import queue
import time
import sys

import cshogi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, ROOT, validate_move, file_sha256

MATE1 = "8k/6G2/7G1/9/9/9/9/9/4K4 b R 1"
# The remote white pawn rules out a non-checking one-move stalemate win.
MATE3 = "8k/5G3/p5G2/9/9/9/9/9/4K4 b R 1"
DECLARATION = "RRBBKGGGG/4SS3/9/9/9/9/9/9/k8 b 18P 1"


def no_bestmove(engine, seconds):
    deadline = time.monotonic() + seconds
    observed = []
    while time.monotonic() < deadline:
        try:
            line = engine.lines.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            break
        assert line is not None, "engine exited while response was held"
        engine.transcript.append(line)
        observed.append(line)
        assert not line.startswith("bestmove "), "premature or duplicate bestmove"
        assert " wdl " not in line, "unexpected WDL output"
    return observed


def go_sfen(engine, sfen, command="nodes 800"):
    engine.send("position sfen " + sfen)
    engine.send("go " + command)
    lines = engine.until(lambda line: line.startswith("bestmove "), 15)
    no_bestmove(engine, .05)
    return lines[-1].split()[1], lines


def verify_mate(sfen, move, lines, distance):
    board = cshogi.Board(sfen)
    selected = validate_move(board, move)
    assert selected is not None
    board.push(selected)
    replies = list(board.legal_moves)
    if distance == 1:
        assert board.is_check() and not replies
    else:
        # MCTS may choose a quiet threat that also forces mate in three.
        # Validate every legal reply, rather than requiring dfpn's check-first PV.
        assert replies
        for reply in replies:
            child = board.copy()
            child.push(reply)
            assert child.mate_move_in_1ply(), "chosen move does not force mate in three"
    certificates = [line for line in lines if f"score mate {distance} " in line]
    assert certificates, f"missing score mate {distance}: {lines[-5:]}"
    for line in certificates:
        assert " pv " in line
        proof = cshogi.Board(sfen)
        moves = line.split(" pv ", 1)[1].split()
        assert len(moves) == distance
        for text in moves:
            proof.push(validate_move(proof, text))
        assert proof.is_check() and not list(proof.legal_moves), "invalid mate PV"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--task-workers", type=int, default=0)
    parser.add_argument("--concurrent-searchers", type=int, default=1)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "protocol.json")
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    engine = Engine(args.engine, args.model, threads=args.threads, slots=args.slots, mate=3,
                    extra_options={"TaskWorkers": args.task_workers,
                                   "MaxConcurrentSearchers": args.concurrent_searchers})
    results = []
    try:
        for sfen, distance in [(MATE1, 1), (MATE3, 3)]:
            move, lines = go_sfen(engine, sfen)
            verify_mate(sfen, move, lines, distance)
            results.append(f"root dfpn mate {distance}")
        assert cshogi.Board(DECLARATION).is_nyugyoku()
        # Declaration and dfpn can complete before classic starts. Neither may
        # bypass infinite/ponder response gating.
        for command, release in [("infinite", "stop"), ("ponder nodes 8", "ponderhit"), ("ponder nodes 8", "stop")]:
            engine.send("position sfen " + DECLARATION)
            engine.send("go " + command)
            no_bestmove(engine, .2)
            start = time.monotonic()
            engine.send(release)
            lines = engine.until(lambda line: line.startswith("bestmove "), 2)
            assert lines[-1] == "bestmove win"
            assert time.monotonic() - start < 2
            no_bestmove(engine, .05)
            results.append(f"early declaration {command} / {release}")
        engine.send("position sfen " + MATE3)
        engine.send("go infinite")
        held = no_bestmove(engine, .2)
        engine.send("stop")
        lines = engine.until(lambda line: line.startswith("bestmove "), 2)
        verify_mate(MATE3, lines[-1].split()[1], held + lines, 3)
        results.append("early root dfpn infinite / stop")

        engine.send("setoption name Mate_Root_Search value 0")
        engine.send("setoption name PV_Mate_Search_Threads value 2")
        engine.send("setoption name PV_Mate_Search_Depth value 3")
        engine.send("setoption name PV_Mate_Search_Nodes value 5000")
        engine.send("isready")
        engine.until(lambda line: line == "readyok", 30)
        for sfen, distance in [(MATE1, 1), (MATE3, 3)]:
            move, lines = go_sfen(engine, sfen)
            verify_mate(sfen, move, lines, distance)
            results.append(f"leaf mate {distance} with PV workers")

        start = cshogi.Board().sfen()
        for command, release in [("infinite", "stop"), ("ponder nodes 8", "ponderhit"), ("ponder nodes 8", "stop")]:
            engine.send("position startpos")
            engine.send("go " + command)
            no_bestmove(engine, .2)
            begin = time.monotonic()
            engine.send(release)
            lines = engine.until(lambda line: line.startswith("bestmove "), 2)
            validate_move(cshogi.Board(), lines[-1].split()[1])
            assert time.monotonic() - begin < 2
            no_bestmove(engine, .05)
            results.append(f"active search {command} / {release}")

        engine.send("setoption name MultiPV value 3")
        _, lines = engine.go([], "nodes 128")
        assert all(any(f" multipv {index} " in line for line in lines) for index in (1, 2, 3))
        results.append("MultiPV 3")

        for iteration in range(20):
            engine.send("position startpos")
            engine.send("go infinite")
            time.sleep(.005)
            engine.send("stop")
            lines = engine.until(lambda line: line.startswith("bestmove "), 2)
            validate_move(cshogi.Board(), lines[-1].split()[1])
            no_bestmove(engine, .01)
        results.append("twenty rapid stops with two PV workers")

        engine.send("setoption name ReuseSubtree value true")
        engine.send("isready")
        engine.until(lambda line: line == "readyok", 30)
        for _ in range(3):
            board = cshogi.Board()
            first, _ = engine.go([], "nodes 128")
            board.push(validate_move(board, first))
            reply = cshogi.move_to_usi(next(iter(board.legal_moves)))
            board.push_usi(reply)
            move, _ = engine.go([first, reply], "nodes 128")
            validate_move(board, move)
        results.append("root advance and rewind with two PV workers")

        for black, white in [(500, 500), (400, 600), (400, 400)]:
            for color, value in [("b", black), ("w", white)]:
                engine.send(f"setoption name Draw_Value_Black value {black}")
                engine.send(f"setoption name Draw_Value_White value {white}")
                engine.send("setoption name Draw_Ply value 1")
                engine.send("isready")
                engine.until(lambda line: line == "readyok", 30)
                sfen = start.replace(" b ", f" {color} ")
                move, lines = go_sfen(engine, sfen, "nodes 64")
                validate_move(cshogi.Board(sfen), move)
                scores = [int(line.split(" score cp ", 1)[1].split()[0]) for line in lines if " score cp " in line and " multipv 1 " in line]
                expected = int(756 * math.log(value / (1000 - value)))
                assert scores and abs(scores[-1] - expected) <= 1, (black, white, color, scores, expected)
                assert not any("score mate" in line for line in lines)
                results.append(f"draw {black}/{white} root {color}")
        # The only white move leaves black a physical one-move mate, but the
        # move-limit rule has already ended the game. Even endpoint draw scores
        # (Q=+/-1) must stay cp scores, rather than being mislabeled as mate.
        capped_mate = "8k/6G2/6pG1/9/9/9/9/9/4K4 w R 1"
        for value in (0, 1000):
            engine.send(f"setoption name Draw_Value_White value {value}")
            engine.send("isready")
            engine.until(lambda line: line == "readyok", 30)
            move, lines = go_sfen(engine, capped_mate, "nodes 64")
            board = cshogi.Board(capped_mate)
            board.push(validate_move(board, move))
            assert board.mate_move_in_1ply(), "fixture must have a physical mate beyond the rule limit"
            assert not any("score mate" in line for line in lines), "rule draw mislabeled as mate"
            expected = -30000 if value == 0 else 30000
            assert any(f"score cp {expected} " in line for line in lines)
        results.append("draw endpoints do not report mate beyond Draw_Ply")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"passed": results, "engine": str(Path(args.engine).resolve()), "model": str(Path(args.model).resolve()),
                                          "engine_sha256": file_sha256(args.engine), "model_sha256": file_sha256(args.model),
                                          "threads": args.threads, "slots": args.slots, "task_workers": args.task_workers,
                                          "max_concurrent_searchers": args.concurrent_searchers}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"passed": len(results), "report": str(args.report)}), flush=True)
    finally:
        engine.close()
        (args.report.parent / (args.report.stem + ".usi.log")).write_text("\n".join(engine.transcript) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
