"""
Run continuous self-play with one USI engine and write all games into one hcpe3 file.

The script uses MultiPV info lines to create hcpe3 MoveVisits records.  USI scores are
interpreted as side-to-move centipawns.  Candidates whose evaluation is much worse
than the best MultiPV line are dropped before writing, and visitNum is derived by a
softmax over the remaining candidate scores.

Example:
    python selfplay_multipv_to_hcpe3.py ./engine ./selfplay.hcpe3 \
        --games 100 --multipv 10 --nodes 1000000 --resign 1000 --draw 256
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from cshogi import dtypeHcp, dtypeMove16, dtypeEval, Board, opponent, move16, BLACK, WHITE, REPETITION_WIN, REPETITION_LOSE, DRAW, BLACK_WIN, WHITE_WIN
from cshogi.usi import Engine


HuffmanCodedPosAndEval3 = np.dtype([
    ("hcp", dtypeHcp),          # start position
    ("moveNum", np.uint16),     # number of moves stored after hcp
    ("result", np.uint8),       # low 2 bits: winner/draw; flags: repetition, nyugyoku, max moves
    ("opponent", np.uint8),     # 0: self-play, 1: black usi, 2: white usi
])
MoveInfo = np.dtype([
    ("selectedMove16", dtypeMove16),
    ("eval", dtypeEval),
    ("candidateNum", np.uint16),
])
MoveVisits = np.dtype([
    ("move16", dtypeMove16),
    ("visitNum", np.uint16),
])

# Same bit convention as csa_to_hcpe3.py.  cshogi turns are BLACK=0, WHITE=1.
RESULT_DRAW = 2
RESULT_REPETITION = 4
RESULT_NYUGYOKU = 8
RESULT_MAX_MOVES = 16

INFO_RE = re.compile(r"^info\b")


@dataclass
class PVInfo:
    multipv: int
    score: int          # side-to-move centipawns; mate is mapped to +/- mate_score
    pv: List[str]


class MultiPVListener:
    """Collects USI output for one search."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.lines: List[str] = []
        self.last_line = ""

    def clear(self) -> None:
        self.lines.clear()
        self.last_line = ""

    def __call__(self, line: str) -> None:
        if self.debug:
            print(line)
        self.last_line = line
        self.lines.append(line)


def parse_options(option_text: str) -> Dict[str, str]:
    """Parse 'name:value,name2:value2'.  Colons inside values are preserved."""
    options: Dict[str, str] = {}
    if not option_text:
        return options
    for kv_text in option_text.split(','):
        if not kv_text:
            continue
        kv = kv_text.split(':', 1)
        if len(kv) != 2:
            raise ValueError(f"bad option format: {kv_text!r}; expected name:value")
        options[kv[0]] = kv[1]
    return options


def parse_usi_info(line: str, mate_score: int) -> Optional[PVInfo]:
    """Parse one USI info line containing score, multipv, and pv."""
    if INFO_RE.match(line) is None:
        return None
    tokens = line.strip().split()
    if "score" not in tokens or "pv" not in tokens:
        return None

    multipv = 1
    score: Optional[int] = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "multipv" and i + 1 < len(tokens):
            try:
                multipv = int(tokens[i + 1])
            except ValueError:
                return None
            i += 2
            continue
        if tok == "score" and i + 2 < len(tokens):
            score_kind = tokens[i + 1]
            score_value = tokens[i + 2]
            if score_kind == "cp":
                try:
                    score = int(score_value)
                except ValueError:
                    return None
            elif score_kind == "mate":
                # mate +N / N means winning mate for side to move; -N losing mate.
                score = -mate_score if score_value.startswith('-') else mate_score
            i += 3
            continue
        if tok == "pv":
            pv = tokens[i + 1:]
            if score is None or len(pv) == 0:
                return None
            return PVInfo(multipv=multipv, score=score, pv=pv)
        i += 1
    return None


def latest_multipv(lines: Sequence[str], board: Board, mate_score: int) -> List[Tuple[int, int]]:
    """
    Return [(move, score), ...] ordered by multipv.
    Invalid or duplicate candidate moves are ignored.
    """
    by_multipv: Dict[int, PVInfo] = {}
    for line in lines:
        info = parse_usi_info(line, mate_score=mate_score)
        if info is not None:
            by_multipv[info.multipv] = info

    candidates: List[Tuple[int, int, int]] = []
    seen_moves = set()
    for mpv in sorted(by_multipv):
        info = by_multipv[mpv]
        if not info.pv:
            continue
        try:
            move = board.move_from_usi(info.pv[0])
        except Exception:
            continue
        if move in seen_moves or not board.is_legal(move):
            continue
        seen_moves.add(move)
        candidates.append((mpv, move, info.score))

    return [(move, score) for _, move, score in sorted(candidates, key=lambda x: x[0])]


def visits_from_scores(scores: Sequence[int], visits_sum: int, temperature: float) -> List[int]:
    """
    Convert side-to-move evaluations to uint16 visit counts.

    Softmax is stable and monotonic.  Every candidate receives at least one visit.
    """
    if not scores:
        return []
    if visits_sum < len(scores):
        visits_sum = len(scores)
    visits_sum = min(visits_sum, 65535)

    if temperature <= 0:
        visits = [1] * len(scores)
        best = max(range(len(scores)), key=lambda i: scores[i])
        visits[best] += visits_sum - len(scores)
        return visits

    max_score = max(scores)
    weights = [math.exp(max(-700.0, min(700.0, (s - max_score) / temperature))) for s in scores]
    total = sum(weights)

    base = [1] * len(scores)
    remaining = visits_sum - len(scores)
    raw = [w / total * remaining for w in weights]
    extra = [int(math.floor(x)) for x in raw]
    visits = [b + e for b, e in zip(base, extra)]

    # Distribute rounding residue to the largest fractional parts.
    residue = visits_sum - sum(visits)
    order = sorted(range(len(scores)), key=lambda i: raw[i] - extra[i], reverse=True)
    for i in order[:residue]:
        visits[i] += 1

    return [min(65535, max(1, v)) for v in visits]


def clamp_eval(score: int) -> int:
    return min(32767, max(-32767, int(score)))


def load_openings(path: Optional[str]) -> List[List[str]]:
    if not path:
        return [[]]
    openings: List[List[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Accept either 'startpos moves ...' or just a sequence of USI moves.
            if line.startswith("startpos moves "):
                moves = line[len("startpos moves "):].split()
            elif line == "startpos":
                moves = []
            else:
                moves = line.split()
            openings.append(moves)
    return openings or [[]]


def apply_opening(
    board: Board,
    opening_moves: Sequence[str],
    max_moves: Optional[int],
) -> Tuple[List[str], defaultdict[int, int]]:
    usi_moves: List[str] = []
    repetition_hash: defaultdict[int, int] = defaultdict(int)
    repetition_hash[board.zobrist_hash()] += 1
    for usi in opening_moves:
        if max_moves is not None and len(usi_moves) >= max_moves:
            break
        move = board.move_from_usi(usi)
        if not board.is_legal(move):
            raise ValueError(f"illegal opening move {usi!r} at ply {len(usi_moves) + 1}")
        board.push(move)
        usi_moves.append(usi)
        repetition_hash[board.zobrist_hash()] += 1
    return usi_moves, repetition_hash


def write_game_hcpe3(
    f,
    start_hcp: np.ndarray,
    result: int,
    records: Sequence[Tuple[int, int, List[Tuple[int, int]]]],
) -> int:
    """Write one game and return number of stored positions."""
    hcpe = np.zeros(1, HuffmanCodedPosAndEval3)
    hcpe["result"] = result
    hcpe["opponent"] = 0
    hcpe["hcp"] = start_hcp
    hcpe["moveNum"] = len(records)
    hcpe.tofile(f)

    position_num = 0
    for selected_move, eval_score, candidates in records:
        move_info = np.zeros(1, MoveInfo)
        move_info["selectedMove16"] = move16(selected_move)
        move_info["eval"] = clamp_eval(eval_score)
        move_info["candidateNum"] = len(candidates)
        move_info.tofile(f)

        for cand_move, visit_num in candidates:
            mv = np.zeros(1, MoveVisits)
            mv["move16"] = move16(cand_move)
            mv["visitNum"] = visit_num
            mv.tofile(f)
        if candidates:
            position_num += 1
    return position_num


def result_code(win: int, repetition: bool = False, nyugyoku: bool = False, max_moves: bool = False) -> int:
    if win == BLACK:
        result = BLACK_WIN
    elif win == WHITE:
        result = WHITE_WIN
    else:
        result = DRAW
    if repetition:
        result |= RESULT_REPETITION
    if nyugyoku:
        result |= RESULT_NYUGYOKU
    if max_moves:
        result |= RESULT_MAX_MOVES
    return result


def build_go_kwargs(args, remain_time: Sequence[Optional[int]]) -> Dict[str, Optional[int]]:
    if args.nodes is not None:
        return {"nodes": args.nodes}
    return {
        "byoyomi": args.byoyomi,
        "btime": remain_time[BLACK],
        "wtime": remain_time[WHITE],
        "binc": args.inc,
        "winc": args.inc,
    }


def play_one_game(
    engine: Engine,
    listener: MultiPVListener,
    openings: Sequence[List[str]],
    game_index: int,
    args,
    out_file,
) -> Tuple[int, int, int]:
    """Return (win, move_count, stored_positions)."""
    board = Board()
    board.reset()

    opening = openings[game_index % len(openings)]
    if args.opening_seed is not None:
        # Deterministic but varied when openings were shuffled before the loop.
        opening = openings[game_index % len(openings)]
    played_opening, repetition_hash = apply_opening(board, opening, args.opening_moves)
    start_hcp = np.zeros(1, dtypeHcp)
    board.to_hcp(start_hcp)
    usi_moves = list(played_opening)

    records: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    remain_time: List[Optional[int]] = [args.time, args.time]

    engine.usinewgame(listener=listener)

    is_repetition = False
    is_nyugyoku = False
    is_max_moves = False
    is_illegal = False
    win = RESULT_DRAW

    while True:
        if board.move_number > args.draw:
            is_max_moves = True
            win = RESULT_DRAW
            break
        if board.is_game_over():
            win = opponent(board.turn)
            break

        engine.position(usi_moves, listener=listener)
        listener.clear()

        import time as _time
        start = _time.perf_counter()
        bestmove, _pondermove = engine.go(listener=listener, **build_go_kwargs(args, remain_time))
        elapsed_ms = math.ceil((_time.perf_counter() - start) * 1000)

        if args.time is not None and args.nodes is None:
            remain_time[board.turn] = (remain_time[board.turn] or 0) + (args.inc or 0) - elapsed_ms
            if remain_time[board.turn] < -1000:
                win = opponent(board.turn)
                break
            remain_time[board.turn] = max(0, remain_time[board.turn])

        candidates = latest_multipv(listener.lines, board, args.mate_score)

        if bestmove == "resign":
            win = opponent(board.turn)
            break
        if bestmove == "win":
            is_nyugyoku = True
            win = board.turn
            break

        try:
            selected_move = board.move_from_usi(bestmove)
        except Exception:
            is_illegal = True
            win = opponent(board.turn)
            break
        if not board.is_legal(selected_move):
            is_illegal = True
            win = opponent(board.turn)
            break

        # Ensure selected move is present as a candidate.  Some engines emit bestmove even
        # when the last info line is incomplete.
        cand_moves = [m for m, _s in candidates]
        if selected_move not in cand_moves:
            fallback_score = candidates[0][1] if candidates else 0
            candidates.insert(0, (selected_move, fallback_score))

        if len(candidates) > args.multipv:
            candidates = candidates[:args.multipv]

        if args.eval_drop_threshold >= 0 and candidates:
            best_score = max(score for _move, score in candidates)
            filtered_candidates = [
                (move, score)
                for move, score in candidates
                if best_score - score <= args.eval_drop_threshold or move == selected_move
            ]
            # selected_move should normally be the best move, but keep this guard so that
            # each MoveInfo always has its selectedMove16 represented among candidates.
            if filtered_candidates:
                candidates = filtered_candidates

        scores = [score for _move, score in candidates]
        visits = visits_from_scores(scores, args.visits_sum, args.temperature)
        candidate_visits = [(move, visit) for (move, _score), visit in zip(candidates, visits)]

        selected_eval = 0
        for move, score in candidates:
            if move == selected_move:
                selected_eval = score
                break

        records.append((selected_move, selected_eval, candidate_visits))

        mover = board.turn
        board.push(selected_move)
        usi_moves.append(bestmove)

        key = board.zobrist_hash()
        repetition_hash[key] += 1
        if repetition_hash[key] >= 4:
            draw_status = board.is_draw()
            if draw_status == REPETITION_WIN:
                win = board.turn
            elif draw_status == REPETITION_LOSE:
                win = opponent(board.turn)
            else:
                win = RESULT_DRAW
            is_repetition = True
            break

        # Optional resignation after the move has been recorded.  The score is side-to-move
        # before pushing, so a negative score means the player to move is losing.
        if args.resign is not None and selected_eval <= -abs(args.resign):
            # selected_eval is from the mover's point of view before the push.
            win = opponent(mover)
            break

    engine.gameover(listener=listener)

    if is_illegal:
        print(f"game {game_index + 1}: illegal bestmove; stored as opponent win")

    stored = write_game_hcpe3(
        out_file,
        start_hcp,
        result_code(win, repetition=is_repetition, nyugyoku=is_nyugyoku, max_moves=is_max_moves),
        records,
    )
    return win, len(records), stored


def main() -> None:
    parser = argparse.ArgumentParser(description="USI self-play with MultiPV -> hcpe3")
    parser.add_argument("engine", help="path to USI engine executable")
    parser.add_argument("hcpe3", help="output hcpe3 file; all games are appended into this one file")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--multipv", type=int, default=10, help="number of PV candidates requested from engine")
    parser.add_argument("--options", type=str, default="", help="additional USI options as name:value,name2:value2")
    parser.add_argument("--nodes", type=int, help="go nodes N; preferred for deterministic self-play data")
    parser.add_argument("--byoyomi", type=int, help="go byoyomi in milliseconds")
    parser.add_argument("--time", type=int, help="initial time per side in milliseconds")
    parser.add_argument("--inc", type=int, default=0, help="increment per move in milliseconds")
    parser.add_argument("--draw", type=int, default=512, help="max move_number before max-move draw flag")
    parser.add_argument("--resign", type=int, help="resign when selected eval <= -resign")
    parser.add_argument("--mate-score", type=int, default=30000, help="cp value used for mate scores")
    parser.add_argument("--visits-sum", type=int, default=65535, help="sum of visitNum over candidates at each position")
    parser.add_argument("--temperature", type=float, default=100.0, help="softmax temperature in centipawns; 0 => all visits to best")
    parser.add_argument("--eval-drop-threshold", type=int, default=500,
                        help="drop MultiPV candidates whose score is worse than the best candidate by more than this cp value; negative disables")
    parser.add_argument("--opening", type=str, help="file with one opening per line: 'startpos moves ...' or USI moves")
    parser.add_argument("--opening-moves", type=int, help="maximum opening plies to apply before recording")
    parser.add_argument("--opening-seed", type=int, help="shuffle openings with this seed")
    parser.add_argument("--append", action="store_true", help="append to hcpe3 instead of overwriting")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.nodes is None and args.byoyomi is None and args.time is None:
        raise ValueError("specify at least one search limit: --nodes, --byoyomi, or --time")
    if args.multipv <= 0:
        raise ValueError("--multipv must be positive")
    if args.visits_sum <= 0:
        raise ValueError("--visits-sum must be positive")

    openings = load_openings(args.opening)
    if args.opening_seed is not None:
        random.seed(args.opening_seed)
    random.shuffle(openings)

    listener = MultiPVListener(debug=args.debug)
    engine = Engine(args.engine, connect=False)
    engine.connect(listener=listener)

    try:
        engine.setoption("MultiPV", str(args.multipv), listener=listener)
        for name, value in parse_options(args.options).items():
            engine.setoption(name, value, listener=listener)
        engine.isready(listener=listener)

        mode = "ab" if args.append else "wb"
        os.makedirs(os.path.dirname(os.path.abspath(args.hcpe3)) or ".", exist_ok=True)
        with open(args.hcpe3, mode) as f:
            wins = [0, 0, 0]
            total_positions = 0
            for game_index in range(args.games):
                win, move_count, stored = play_one_game(engine, listener, openings, game_index, args, f)
                f.flush()
                os.fsync(f.fileno())
                total_positions += stored
                if win == BLACK:
                    wins[BLACK] += 1
                    result_text = "black win"
                elif win == WHITE:
                    wins[WHITE] += 1
                    result_text = "white win"
                else:
                    wins[2] += 1
                    result_text = "draw"
                print(f"game {game_index + 1}/{args.games}: {result_text}, moves={move_count}, positions={stored}")
            print(f"summary: black={wins[BLACK]}, white={wins[WHITE]}, draw={wins[2]}, positions={total_positions}")
    finally:
        engine.quit(listener=listener)


if __name__ == "__main__":
    main()
