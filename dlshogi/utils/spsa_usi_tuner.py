"""
SPSA-based USI parameter tuner with direct updates and optional baseline monitoring.

Usage example
-------------
python spsa_usi_tuner.py   --baseline ./my_baseline_engine   --candidate ./my_candidate_engine   --options-baseline "Threads:1,Hash:64"   --options-candidate "Threads:1,Hash:64"   --params "C_init:0~200:144:5:1:int,C_base:20000~50000:28288:1000:0.5:int"   --openings openings.sfen   --iterations 50   --sets-per-iter 3   --repeat-per-pair 2   --workers 3   --byoyomi 1000   --draw-moves 320   --monitor-every 10   --monitor-games 12

The default update mode is direct (theta+ vs theta-). Baseline monitoring is
disabled unless --monitor-every K is supplied; it evaluates the updated theta.
Use --update-mode base to optimize only the score difference against a fixed base.
In direct mode, OpenBench's pairs_per corresponds to sets_per_iter * repeat_per_pair.
Scores are summed, not averaged over pairs. Compared with the former combined
update, direct updates have about half the expected magnitude under a local
equal-strength model. Doubling r_end approximately preserves that update scale.
Engines stay resident per worker and are closed when the worker exits. Search
options are reapplied before each game; changes to dlshogi initialization-only
options restart the affected process. A distinct baseline executable is retained
between monitors as an additional process per worker that uses it.

Parameter spec format (for --params, comma separated list):
  name:min~max:start[:c_end][:r_end][:type]

  - name     : USI option name to tune
  - min~max  : hard range (integers recommended for USI integer options)
  - start    : starting value (inside [min,max])
  - c_end    : (optional) target perturbation scale at the final iteration (default: 1; int uses at least 0.5)
  - r_end    : (optional) target r = a/c^2 at the final iteration (default: 0.5)
  - type     : (optional) "int" (default) or "float"
"""

import argparse
import math
import random
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from cshogi import Board, BLACK, WHITE, REPETITION_WIN, REPETITION_LOSE
from cshogi.usi import Engine


# --- SPSA structures ------------------------------------------------------------

@dataclass
class SpsaParam:
    name: str
    vmin: float
    vmax: float
    value: float
    c_end: float = 1.0
    r_end: float = 0.5
    is_int: bool = True

    # derived (set in setup)
    c0: float = 0.0
    a_end: float = 0.0
    a0: float = 0.0

    def clip(self, x: float) -> float:
        return float(max(self.vmin, min(self.vmax, x)))

    def clip_and_cast(self, x: float, u: Optional[float] = None) -> float:
        x = self.clip(x)
        if self.is_int:
            # Keep integer options within the range, including fractional bounds.
            lower, upper = math.ceil(self.vmin), math.floor(self.vmax)
            if lower > upper:
                raise ValueError(f"No integer in range for parameter '{self.name}'")
            x = max(lower, min(upper, x))
            if u is not None:
                return math.floor(x + u)
            return int(round(x))
        return x

    def as_usi_value(self) -> str:
        return str(self.clip_and_cast(self.value))


@dataclass
class SpsaConfig:
    alpha: float = 0.602
    gamma: float = 0.101
    A_ratio: float = 0.1
    iterations: int = 250
    sets_per_iter: int = 1
    repeat_per_pair: int = 1  # 2 * repeat games/set in direct mode; twice that in base mode
    # computed later
    A: float = 0.0

    def setup(self):
        self.A = self.A_ratio * self.iterations


# --- Parsing --------------------------------------------------------------------

def parse_options_templates(s: str) -> List[Tuple[str, str]]:
    """
    Parse "k1:v1,k2:v2" into list of (key_template, value_template).
    """
    def split_key_value(kv_str: str) -> Tuple[str, str]:
        brace_level = 0
        for i, char in enumerate(kv_str):
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            elif char == ':' and brace_level == 0:
                return kv_str[:i].strip(), kv_str[i+1:].strip()
        raise ValueError(f"Invalid options item: '{kv_str}' (expected 'key:value')")

    out: List[Tuple[str, str]] = []
    s = (s or "").strip()
    if not s:
        return out
    for kv_str in s.split(","):
        if not kv_str:
            continue
        k, v = split_key_value(kv_str)
        out.append((k, v))
    return out


def parse_params(s: str) -> List[SpsaParam]:
    """
    Parse --params string:
      name:min~max:start[:c_end][:r_end][:type]
    Multiple params separated by commas.
    """
    params: List[SpsaParam] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 3:
            raise ValueError(f"Param spec needs at least name:min~max:start: {item}")
        name = parts[0]
        rng = parts[1]
        start = float(parts[2])
        c_end = float(parts[3]) if len(parts) >= 4 and parts[3] != "" else 1.0
        r_end = float(parts[4]) if len(parts) >= 5 and parts[4] != "" else 0.5
        typ = (parts[5] if len(parts) >= 6 else "int").strip().lower()
        vmin, vmax = [float(x) for x in rng.split("~")]
        is_int = typ != "float"
        p = SpsaParam(name=name, vmin=vmin, vmax=vmax, value=start, c_end=c_end, r_end=r_end, is_int=is_int)
        params.append(p)
    return params


# --- USI game runner ------------------------------------------------------------
# This is a trimmed version oriented to "position sfen <SFEN>" starts, byoyomi clock only.

class GameRunner:
    def __init__(self,
                 byoyomi_ms: int = 1000,
                 draw_moves: int = 320,
                 debug: bool = False):
        self.byoyomi_ms = byoyomi_ms
        self.draw_moves = draw_moves
        self.debug = debug
        # Each worker owns its processes; no USI stream is shared across threads.
        self._local = threading.local()

    def _engine_cache(self):
        if not hasattr(self._local, 'engines'):
            self._local.engines = {}
        return self._local.engines

    @staticmethod
    def _initialization_options(options: Dict[str, str]) -> Dict[str, str]:
        # usi/main.cpp reads these only inside the first-isready initialization.
        prefixes = ('DNN_Model', 'DNN_Batch_Size', 'UCT_Threads', 'Multi_Ponder', 'PV_Mate_Search_')
        names = {'UCT_NodeLimit', 'DNN_Cache_Size', 'DfPn_Hash', 'Mate_Root_Search'}
        return {k: v for k, v in options.items() if k in names or k.startswith(prefixes)}

    @staticmethod
    def _quit_engine(eng: Engine):
        try:
            eng.quit()
        except Exception:
            pass

    def close(self):
        """Quit all processes owned by the calling worker (also safe after failure)."""
        cache = self._engine_cache()
        engines = [eng for eng, _ in cache.values()]
        cache.clear()
        for eng in engines:
            self._quit_engine(eng)

    def _connect_and_setup(self, cmd: str, options: Dict[str, str], slot: int = 0) -> Engine:
        cache = self._engine_cache()
        # Slots distinguish two instances of the same candidate executable.
        # Keeping commands separate also retains the baseline between monitors.
        key = (slot, cmd)
        if key in cache:
            eng, previous = cache[key]
            if (previous.keys() - options.keys() or
                    self._initialization_options(previous) != self._initialization_options(options)):
                # Removed options need their defaults restored. Initialization-only
                # options cannot be reapplied by a subsequent dlshogi isready.
                del cache[key]
                self._quit_engine(eng)
        if key not in cache:
            eng = Engine(cmd, connect=False, debug=self.debug)
            cache[key] = (eng, {})  # Include partially connected processes in cleanup.
            eng.connect()
        else:
            eng, _ = cache[key]
        for k, v in options.items():
            eng.setoption(k, v)
        # dlshogi reuses the model, resets game state and reapplies search options.
        eng.isready()
        eng.usinewgame()
        cache[key] = (eng, dict(options))
        return eng

    def _play_single(self, sfen: str,
                     black_cmd: str, white_cmd: str,
                     black_opts: Dict[str, str], white_opts: Dict[str, str],
                     black_slot: int = 0, white_slot: int = 1) -> int:
        try:
            b_eng = self._connect_and_setup(black_cmd, black_opts, black_slot)
            w_eng = self._connect_and_setup(white_cmd, white_opts, white_slot)
            winner = self._play_game(sfen, b_eng, w_eng)
            # USI results are from each engine's own perspective.
            b_eng.gameover('win' if winner > 0 else 'lose' if winner < 0 else 'draw')
            w_eng.gameover('lose' if winner > 0 else 'win' if winner < 0 else 'draw')
            return winner
        except BaseException:
            # A broken protocol stream must not be reused by another match.
            self.close()
            raise

    def _play_game(self, sfen: str, b_eng: Engine, w_eng: Engine) -> int:
        """
        Play one game from an SFEN start. Returns result: +1 (black win), -1 (white win), 0 (draw).
        """
        board = Board()

        # Keep local record of moves to pass "position ... moves" to pondering if needed.
        usi_moves: List[str] = []
        repetition_hash = defaultdict(int)

        start_pos = sfen.strip().split(" moves ", 1)
        start_sfen = start_pos[0]
        if start_sfen == "startpos":
            pass
        elif start_sfen.startswith("sfen "):
            board.set_sfen(start_sfen[5:])
        else:
            raise ValueError(f"Invalid SFEN start position: '{start_sfen}'")
        repetition_hash[board.zobrist_hash()] += 1
        if len(start_pos) == 2:
            moves_str = start_pos[1]
            for mv_usi in moves_str.strip().split():
                board.push_usi(mv_usi)
                usi_moves.append(mv_usi)
                repetition_hash[board.zobrist_hash()] += 1

        is_game_over = False
        winner = 0  # 1 black, -1 white, 0 draw

        while not is_game_over:
            # Draw by move limit
            if board.move_number > self.draw_moves:
                winner = 0
                break

            turn = board.turn  # 0=BLACK, 1=WHITE
            eng = b_eng if turn == BLACK else w_eng

            # Position with current moves for the side to move
            eng.position(sfen=start_sfen, moves=usi_moves)

            # Search
            bestmove, ponder = eng.go(byoyomi=self.byoyomi_ms)

            if bestmove == "resign":
                winner = -1 if turn == BLACK else 1
                is_game_over = True
                break
            elif bestmove == "win":
                # nyugyoku declaration win
                # Trust the USI engine's declaration without validating nyugyoku conditions.
                winner = 1 if turn == BLACK else -1
                is_game_over = True
                break
            else:
                # Make move on board
                try:
                    mv = board.move_from_usi(bestmove)
                except Exception:
                    # illegal notation => resign
                    winner = -1 if turn == BLACK else 1
                    is_game_over = True
                    break

                if board.is_legal(mv):
                    board.push(mv)
                    usi_moves.append(bestmove)
                    key = board.zobrist_hash()
                    repetition_hash[key] += 1
                    if repetition_hash[key] == 4:
                        # Fourfold repetition handling
                        is_draw = board.is_draw()
                        # Repetition results are for the side to move after push (opposite of turn).
                        if is_draw == REPETITION_WIN:
                            winner = -1 if turn == BLACK else 1
                        elif is_draw == REPETITION_LOSE:
                            winner = 1 if turn == BLACK else -1
                        else:
                            winner = 0
                        is_game_over = True
                        break
                else:
                    # illegal move -> opponent wins
                    winner = -1 if turn == BLACK else 1
                    is_game_over = True
                    break

            if board.is_game_over():
                # result by checkmate/toryo etc.
                # cshogi Board doesn't directly give winner here; decide by side to move having no legal move.
                # As a fallback consider last mover winner.
                last_mover = 1 if turn == BLACK else -1
                winner = last_mover
                is_game_over = True
                break

        return winner

    def play_pair_color_swapped(self, sfen: str,
                                a_cmd: str, b_cmd: str,
                                a_opts: Dict[str, str], b_opts: Dict[str, str],
                                repeat: int = 1) -> Tuple[int, int, int]:
        """
        Play 2*repeat games (A vs B with colors swapped each time).
        Returns (A_wins, B_wins, draws).
        """
        A_w, B_w, D = 0, 0, 0
        for r in range(repeat):
            # game 1: A as black
            res = self._play_single(sfen, a_cmd, b_cmd, a_opts, b_opts)
            if res > 0: A_w += 1
            elif res < 0: B_w += 1
            else: D += 1
            # game 2: A as white (swap)
            res = self._play_single(sfen, b_cmd, a_cmd, b_opts, a_opts, black_slot=1, white_slot=0)
            # invert perspective
            if res > 0: B_w += 1  # black (B) won
            elif res < 0: A_w += 1
            else: D += 1
        return A_w, B_w, D


# --- SPSA core (Fishtest-style with per-param c_end, r_end schedules) ------------
# Design per "SPSA in Fishtest.md" and simplified reference.

class SpsaTuner:
    def __init__(self,
                 baseline_cmd: Optional[str],
                 candidate_cmd: str,
                 baseline_options: List[Tuple[str, str]],
                 candidate_base_options: List[Tuple[str, str]],
                 params: List[SpsaParam],
                 openings: List[str],
                 spsa_cfg: SpsaConfig,
                 byoyomi_ms: int = 1000,
                 draw_moves: int = 320,
                 workers: int = 1,
                 update_mode: str = "direct",
                 random_seed: Optional[int] = None,
                 debug: bool = False,
                 monitor_every: int = 0,
                 monitor_games: int = 2):
        if update_mode not in ("direct", "base"):
            raise ValueError("update_mode must be 'direct' or 'base'")
        if spsa_cfg.iterations <= 0 or spsa_cfg.sets_per_iter <= 0 or spsa_cfg.repeat_per_pair <= 0:
            raise ValueError("iterations, sets_per_iter and repeat_per_pair must be positive")
        if monitor_every < 0 or monitor_games <= 0 or monitor_games % 2:
            raise ValueError("monitor_every must be nonnegative and monitor_games must be positive and even")
        if (update_mode == "base" or monitor_every > 0) and not baseline_cmd:
            raise ValueError("A baseline engine is required for base updates or monitoring")
        if not openings:
            raise ValueError("At least one opening is required")
        self.base_cmd = baseline_cmd
        self.cand_cmd = candidate_cmd
        self.base_opts_templates = baseline_options
        self.cand_base_opts_templates = candidate_base_options
        self.params = params
        self.openings = openings
        self.cfg = spsa_cfg
        self.runner = GameRunner(byoyomi_ms=byoyomi_ms, draw_moves=draw_moves, debug=debug)
        self.workers = max(1, workers)
        self.update_mode = update_mode
        self.monitor_every = monitor_every
        self.monitor_games = monitor_games
        self.rnd = random.Random(random_seed)
        # Monitoring must not change future perturbations or training openings.
        self.monitor_rnd = random.Random(None if random_seed is None else f"{random_seed}-monitor")
        self.debug = debug
        self._iter = 0
        self._lock = threading.Lock()

        self.cfg.setup()
        # derive a0, c0 per param (end-anchored schedules) per OpenBench-like code
        for p in self.params:
            p.c0 = p.c_end * (self.cfg.iterations ** self.cfg.gamma)
            p.a_end = p.r_end * (p.c_end ** 2)
            p.a0 = p.a_end * ((self.cfg.A + self.cfg.iterations) ** self.cfg.alpha)

        # Setup worker threads
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_threads = []
        for i in range(self.workers):
            t = threading.Thread(target=self._worker_loop, args=(i,))
            t.daemon = True
            self.worker_threads.append(t)
            t.start()

    def _worker_loop(self, worker_id):
        try:
            while True:
                task_func = self.task_queue.get()
                if task_func is None:
                    break
                try:
                    result = task_func(worker_id)
                    self.result_queue.put(result)
                except Exception as e:
                    self.result_queue.put(e)
        finally:
            self.runner.close()

    # schedule accessors
    def _current_c(self, p: SpsaParam, t: int) -> float:
        c = p.c0 / (t ** self.cfg.gamma)
        # Keep shared-rounding perturbations distinct away from clipped bounds.
        return max(c, 0.5) if p.is_int else c

    def _current_a(self, p: SpsaParam, t: int) -> float:
        return p.a0 / ((self.cfg.A + t) ** self.cfg.alpha)

    def _current_r(self, p: SpsaParam, t: int) -> float:
        c = self._current_c(p, t)
        a = self._current_a(p, t)
        if c == 0:
            return 0.0
        return a / (c ** 2)

    def _sample_opening(self) -> str:
        return self.rnd.choice(self.openings).strip()

    def _build_options_from_templates(self, templates: List[Tuple[str, str]], worker_id: int) -> Dict[str, str]:
        opts: Dict[str, str] = {}
        for key_t, val_t in templates:
            try:
                key = eval(f'fr"{key_t}"', {'id': worker_id})
                val = eval(f'fr"{val_t}"', {'id': worker_id})
            except:
                key = key_t
                val = val_t
            opts[key] = val
        return opts

    def _params_to_options(self, values: Dict[str, float]) -> Dict[str, str]:
        opts: Dict[str, str] = {}
        for p in self.params:
            val = p.clip_and_cast(values[p.name])
            opts[p.name] = str(int(val) if p.is_int else float(val))
        return opts

    def _current_theta(self) -> Dict[str, float]:
        return {p.name: p.value for p in self.params}

    def _apply_update(self, deltas: Dict[str, float]):
        for p in self.params:
            # Keep fractional updates even for integer engine options.
            p.value = p.clip(p.value + deltas.get(p.name, 0.0))

    def _task_plus_base(self, sfen, cand_cmd, base_cmd, cand_templates, base_templates, theta, repeat, worker_id):
        cand_opts = self._build_options_from_templates(cand_templates, worker_id)
        cand_opts.update(self._params_to_options(theta))
        base_opts = self._build_options_from_templates(base_templates, worker_id)
        scores = self.runner.play_pair_color_swapped(sfen, cand_cmd, base_cmd, cand_opts, base_opts, repeat)
        return ('plus_base', scores)

    def _task_minus_base(self, sfen, cand_cmd, base_cmd, cand_templates, base_templates, theta, repeat, worker_id):
        cand_opts = self._build_options_from_templates(cand_templates, worker_id)
        cand_opts.update(self._params_to_options(theta))
        base_opts = self._build_options_from_templates(base_templates, worker_id)
        scores = self.runner.play_pair_color_swapped(sfen, cand_cmd, base_cmd, cand_opts, base_opts, repeat)
        return ('minus_base', scores)

    def _task_plus_minus(self, sfen, cand_cmd1, cand_cmd2, cand_templates1, cand_templates2, theta1, theta2, repeat, worker_id):
        cand_opts1 = self._build_options_from_templates(cand_templates1, worker_id)
        cand_opts1.update(self._params_to_options(theta1))
        cand_opts2 = self._build_options_from_templates(cand_templates2, worker_id)
        cand_opts2.update(self._params_to_options(theta2))
        scores = self.runner.play_pair_color_swapped(sfen, cand_cmd1, cand_cmd2, cand_opts1, cand_opts2, repeat)
        return ('plus_minus', scores)

    def _collect_scores(self, task_count: int) -> Dict[str, List[int]]:
        scores = defaultdict(lambda: [0, 0, 0])
        error = None
        for _ in range(task_count):
            result = self.result_queue.get()
            if isinstance(result, Exception):
                error = result if error is None else error
                continue
            label, tri = result
            for i, count in enumerate(tri):
                scores[label][i] += count
        if error is not None:
            raise error
        return scores

    def _monitor(self, t: int):
        # Evaluate the updated central theta, rounded once without perturbations.
        theta = {p.name: p.clip_and_cast(p.value) for p in self.params}
        pairs = self.monitor_games // 2
        for _ in range(pairs):
            sfen = self.monitor_rnd.choice(self.openings).strip()
            self.task_queue.put(lambda wid, sfen=sfen: self._task_plus_base(
                sfen, self.cand_cmd, self.base_cmd, self.cand_base_opts_templates,
                self.base_opts_templates, theta, 1, wid))
        wins, losses, draws = self._collect_scores(pairs)['plus_base']
        score = (wins + 0.5 * draws) / self.monitor_games
        print(f"[Monitor {t:3d}] games={self.monitor_games}, theta={theta}")
        print(f"  theta vs base : W-L-D = {wins}-{losses}-{draws}, score={score:.1%}")

    def _iteration_once(self, t: int) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Run one SPSA iteration:
          - sample delta in {+1,-1}^d
          - construct theta± = theta ± c_t * delta (with clipping)
          - run N sets in parallel, using either direct or fixed-base matches
          - aggregate the selected mode's score into magnitude
          - compute update Δθ = r_t * c_t * magnitude * delta
        Returns (theta, theta+, theta-).
        """
        theta = self._current_theta()
        # sample delta signs
        delta_signs: Dict[str, int] = {p.name: (1 if self.rnd.getrandbits(1) else -1) for p in self.params}

        # construct theta+ / theta-
        c_t: Dict[str, float] = {p.name: self._current_c(p, t) for p in self.params}
        r_t: Dict[str, float] = {p.name: self._current_r(p, t) for p in self.params}

        theta_plus: Dict[str, float] = {}
        theta_minus: Dict[str, float] = {}

        # Sample integer options once per iteration, before dispatching workers.
        # Reuse these values in every match involving the same perturbed engine.
        for p in self.params:
            # A shared uniform sample preserves the ordering of the perturbations.
            u = self.rnd.random() if p.is_int else None
            theta_plus[p.name]  = p.clip_and_cast(theta[p.name] + c_t[p.name] * delta_signs[p.name], u)
            theta_minus[p.name] = p.clip_and_cast(theta[p.name] - c_t[p.name] * delta_signs[p.name], u)

        # Prepare options dicts for engines:
        # candidate+ options = candidate_base_options + param overrides
        # Options are built dynamically in tasks

        # Submit sets to worker threads
        for _ in range(self.cfg.sets_per_iter):
            sfen = self._sample_opening()
            if self.update_mode == "direct":
                self.task_queue.put(lambda wid, sfen=sfen: self._task_plus_minus(sfen, self.cand_cmd, self.cand_cmd, self.cand_base_opts_templates, self.cand_base_opts_templates, theta_plus, theta_minus, self.cfg.repeat_per_pair, wid))
            else:
                self.task_queue.put(lambda wid, sfen=sfen: self._task_plus_base(sfen, self.cand_cmd, self.base_cmd, self.cand_base_opts_templates, self.base_opts_templates, theta_plus, self.cfg.repeat_per_pair, wid))
                self.task_queue.put(lambda wid, sfen=sfen: self._task_minus_base(sfen, self.cand_cmd, self.base_cmd, self.cand_base_opts_templates, self.base_opts_templates, theta_minus, self.cfg.repeat_per_pair, wid))

        matches_per_set = 1 if self.update_mode == "direct" else 2
        scores = self._collect_scores(self.cfg.sets_per_iter * matches_per_set)

        # Convert to signed scores (win=+1, draw=0, loss=-1).
        def signed_score(tri: List[int]) -> float:
            W, L, _ = tri
            return float(W) - float(L)

        if self.update_mode == "direct":
            magnitude = signed_score(scores['plus_minus'])
        else:
            magnitude = signed_score(scores['plus_base']) - signed_score(scores['minus_base'])

        # Parameter update: Δθ_i = r_i(t) * c_i(t) * magnitude * delta_i
        # (Fishtest-style: r = a/c^2; see docs)
        updates: Dict[str, float] = {}
        # Sum results without pair-count normalization, as in OpenBench.
        for p in self.params:
            updates[p.name] = (r_t[p.name] * c_t[p.name] *
                               magnitude * float(delta_signs[p.name]))

        # Apply update with clipping
        self._apply_update(updates)

        # Logging
        total_sets = self.cfg.sets_per_iter
        total_games = total_sets * matches_per_set * 2 * self.cfg.repeat_per_pair
        print(f"[Iter {t:3d}] mode={self.update_mode}, sets={total_sets}, games={total_games}")
        print(f"  theta+    : " + ", ".join(f"{k}={theta_plus[k]}" for k in theta_plus))
        print(f"  theta-    : " + ", ".join(f"{k}={theta_minus[k]}" for k in theta_minus))
        labels = [('plus_minus', '(+) vs (-)')] if self.update_mode == "direct" else [
            ('plus_base', '(+) vs base'), ('minus_base', '(-) vs base')]
        for key, label in labels:
            w, l, d = scores[key]
            print(f"  {label} : W-L-D = {w}-{l}-{d} (S={signed_score(scores[key]):.1f})")
        print(f"  magnitude     : {magnitude:+.2f}")
        print(f"  updates       : " + ", ".join(f"{k}={updates[k]:+.3f}" for k in updates))
        print(f"  updated theta : " + ", ".join(f"{p.name}={p.value}" for p in self.params))

        return theta, theta_plus, theta_minus

    def run(self, start_iter: int = 1, end_iter: Optional[int] = None):
        end_iter = end_iter or self.cfg.iterations
        try:
            for t in range(start_iter, end_iter + 1):
                self._iteration_once(t)
                if self.monitor_every and t % self.monitor_every == 0:
                    self._monitor(t)
        finally:
            # Discard unstarted matches on interruption; running matches still finish.
            while True:
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            for _ in range(self.workers):
                self.task_queue.put(None)
            for worker in self.worker_threads:
                worker.join()


# --- CLI ------------------------------------------------------------------------

def load_openings_sfen(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    # allow lines that might include "sfen " prefix; store as-is
    return lines

def main():
    ap = argparse.ArgumentParser(description="SPSA (Fishtest style) USI parameter tuner")
    ap.add_argument("--baseline", help="Path to baseline engine (required for base updates or monitoring)")
    ap.add_argument("--candidate", required=True, help="Path to candidate engine binary (used for +/- perturbations)")
    ap.add_argument("--options-baseline", default="", help='USI options for baseline: "Name:Value,Name:Value,..."')
    ap.add_argument("--options-candidate", default="", help='Base USI options for candidate: "Name:Value,..." (tuned params override these)')
    ap.add_argument("--params", required=True, help='Param specs: name:min~max:start[:c_end][:r_end][:type], comma-separated')
    ap.add_argument("--openings", required=True, help="SFEN file with one position per line")
    ap.add_argument("--iterations", type=int, default=250, help="Total SPSA iterations")
    ap.add_argument("--sets-per-iter", type=int, default=1, help="Sets per iteration; recommend >= workers to keep workers busy")
    ap.add_argument("--repeat-per-pair", type=int, default=1, help="Color-swapped pairs per match: direct mode uses 2*repeat games/set, base mode 4*repeat")
    ap.add_argument("--update-mode", choices=("direct", "base"), default="direct", help="Update from theta+ vs theta- (default), or their score difference against baseline")
    ap.add_argument("--monitor-every", type=int, default=0, help="Monitor updated theta against baseline every K iterations (0: disabled, default)")
    ap.add_argument("--monitor-games", type=int, default=2, help="Games per monitor, positive and even (default: 2); recommend at least several dozen for meaningful scores and >= 2*workers to use all workers; never used for updates")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers; budget GPU memory for 2-3 models per worker (candidate pair plus a distinct monitoring baseline)")
    ap.add_argument("--alpha", type=float, default=0.602, help="SPSA alpha (learning-rate decay)")
    ap.add_argument("--gamma", type=float, default=0.101, help="SPSA gamma (perturbation decay)")
    ap.add_argument("--A-ratio", type=float, default=0.1, dest="A_ratio", help="Stabilizer A as a fraction of iterations")
    ap.add_argument("--byoyomi", type=int, default=1000, help="Byoyomi in milliseconds")
    ap.add_argument("--draw-moves", type=int, default=320, help="Declare draw after this many moves")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.iterations <= 0 or args.sets_per_iter <= 0 or args.repeat_per_pair <= 0:
        ap.error("--iterations, --sets-per-iter and --repeat-per-pair must be positive")
    if args.monitor_every < 0 or args.monitor_games <= 0 or args.monitor_games % 2:
        ap.error("--monitor-every must be nonnegative; --monitor-games must be positive and even")
    if (args.update_mode == "base" or args.monitor_every > 0) and not args.baseline:
        ap.error("--baseline is required for --update-mode base or enabled monitoring")

    # Build configs
    spsa_cfg = SpsaConfig(alpha=args.alpha, gamma=args.gamma, A_ratio=args.A_ratio,
                          iterations=args.iterations, sets_per_iter=args.sets_per_iter,
                          repeat_per_pair=args.repeat_per_pair)
    params = parse_params(args.params)
    opts_base = parse_options_templates(args.options_baseline)
    opts_cand = parse_options_templates(args.options_candidate)
    openings = load_openings_sfen(args.openings)

    print(f"Loaded {len(openings)} openings from {args.openings}.")
    print("Parameters:")
    for p in params:
        print(f"  {p.name}: range=({p.vmin},{p.vmax}), start={p.value}, c_end={p.c_end}, r_end={p.r_end}, int={p.is_int}")

    tuner = SpsaTuner(
        baseline_cmd=args.baseline,
        candidate_cmd=args.candidate,
        baseline_options=opts_base,
        candidate_base_options=opts_cand,
        params=params,
        openings=openings,
        spsa_cfg=spsa_cfg,
        byoyomi_ms=args.byoyomi,
        draw_moves=args.draw_moves,
        workers=args.workers,
        update_mode=args.update_mode,
        monitor_every=args.monitor_every,
        monitor_games=args.monitor_games,
        random_seed=args.seed,
        debug=args.debug
    )
    tuner.run()

    print("Final tuned parameters:")
    for p in tuner.params:
        print(f"  {p.name}: {p.value}")


if __name__ == "__main__":
    main()
