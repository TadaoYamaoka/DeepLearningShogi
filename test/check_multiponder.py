"""Exercise ten actual auxiliary-engine hits and ten misses, then verify exit."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys

import cshogi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, ROOT, validate_move, file_sha256


def child_handles(parent):
    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("exe", wintypes.WCHAR * 260)]
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    api.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = api.CreateToolhelp32Snapshot(2, 0)
    assert snapshot != ctypes.c_void_p(-1).value
    handles = []
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(entry)
        valid = api.Process32FirstW(snapshot, ctypes.byref(entry))
        while valid:
            if entry.parent == parent:
                handle = api.OpenProcess(0x100000, False, entry.pid)
                assert handle, ctypes.get_last_error()
                handles.append((entry.pid, handle))
            valid = api.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        api.CloseHandle(snapshot)
    return api, handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--auxiliary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "multiponder.json")
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    model = str(Path(args.model).resolve())
    auxiliary = str(Path(args.auxiliary).resolve())
    auxiliary_options = ",".join(f"{key}:{value}" for key, value in {
        "DNN_Model": model, "Threads": 1, "DNN_Slots": 1, "TaskWorkers": 0, "DNN_Batch_Size": 128,
        "Mate_Root_Search": 0, "DfPn_Hash": 64, "Resign_Threshold": 0,
        "USI_Ponder": "true", "Stochastic_Ponder": "false", "Time_Margin": 0}.items())
    options = {"USI_Ponder": "true", "Stochastic_Ponder": "true", "Multi_Ponder": 2,
               "Multi_Ponder_Engine1": auxiliary, "Multi_Ponder_Engine2": auxiliary,
               "Multi_Ponder_Engine1_Options": auxiliary_options, "Multi_Ponder_Engine2_Options": auxiliary_options}
    engine = Engine(args.engine, args.model, extra_options=options)
    api, handles = child_handles(engine.process.pid)
    assert len(handles) == 2, f"expected two auxiliary engines, got {len(handles)}"
    results = []
    try:
        for iteration in range(20):
            board = cshogi.Board()
            move, _ = engine.go([], "nodes 64")
            board.push(validate_move(board, move))
            lines = engine.until(lambda line: line.startswith("info string multiponder "), 10)
            fields = lines[-1].split()[3:]
            assert len(fields) == 4, fields
            candidates = [fields[1], fields[3]]
            for candidate in candidates:
                validate_move(board, candidate)
            hit = iteration % 2 == 0
            reply = candidates[0] if hit else next(cshogi.move_to_usi(value) for value in board.legal_moves if cshogi.move_to_usi(value) not in candidates)
            board.push_usi(reply)
            best, lines = engine.go([move, reply], "nodes 64")
            validate_move(board, best)
            observed = any("ponderhit" in line and "multiponder" in line for line in lines)
            assert observed == hit, (iteration, hit, lines)
            engine.send("stop")
            engine.send("isready")
            tail = engine.until(lambda line: line == "readyok", 30)
            assert not any(line.startswith("bestmove ") for line in tail), "duplicate bestmove"
            results.append({"iteration": iteration, "hit": hit, "own": move, "reply": reply, "best": best})
            print(f"multiponder {iteration + 1}/20: {'hit' if hit else 'miss'}", flush=True)
        engine.go([], "nodes 64")
        engine.until(lambda line: line.startswith("info string multiponder "), 10)
        # Finally quit with both auxiliary ponder searches still running.
    finally:
        engine.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.with_suffix(".usi.log").write_text("\n".join(engine.transcript) + "\n", encoding="utf-8")
    try:
        for pid, handle in handles:
            assert api.WaitForSingleObject(handle, 10000) == 0, f"auxiliary process {pid} still alive"
    finally:
        for _, handle in handles:
            api.CloseHandle(handle)
    args.report.write_text(json.dumps({"hits": 10, "misses": 10, "quit_during_ponder": True,
                                      "children_exited": True, "engine_sha256": file_sha256(args.engine),
                                      "auxiliary_sha256": file_sha256(args.auxiliary),
                                      "model_sha256": file_sha256(args.model), "records": results}, indent=2) + "\n", encoding="utf-8")
    print("multiponder passed: ten hits, ten misses, no surviving children", flush=True)


if __name__ == "__main__":
    main()
