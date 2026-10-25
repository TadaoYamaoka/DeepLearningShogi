"""Generate a tiny book, validate its entries, then read it through USI."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

import cshogi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, ROOT, validate_move, file_sha256


def probe(engine):
    engine.send("position startpos")
    engine.send("nnprobe")
    line = engine.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1]
    fields = line.split()
    return float(fields[4]), dict((move, float(value)) for move, value in
                                 (field.split(":") for field in fields[5:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="MakeBook configuration")
    parser.add_argument("--reader", required=True, help="Release configuration with BOOK_POLICY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/final/makebook.json")
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    work = args.report.parent / (args.report.stem + "_data")
    work.mkdir(parents=True, exist_ok=True)
    empty, output = work / "empty.bin", work / "generated.bin"
    empty.write_bytes(b"")
    if output.exists():
        output.unlink()
    writer = Engine(args.engine, args.model, extra_options={
        "Make_Book_Color": "black", "Use_Interruption": "false", "Save_Book_Interval": 1})
    try:
        writer.send(f"make_book {empty.resolve()} {output.resolve()} 128 1")
        writer.until(lambda line: line.startswith("merged entries\t"), 60)
    finally:
        writer.close()
        args.report.with_suffix(".usi.log").write_text("\n".join(writer.transcript) + "\n", encoding="utf-8")
    raw = output.read_bytes()
    assert raw and len(raw) % 16 == 0, "BookEntry layout mismatch"
    entries = list(struct.iter_unpack("<QHHi", raw))
    board = cshogi.Board()
    key = board.book_key()
    legal = {cshogi.move16(move): cshogi.move_to_usi(move) for move in board.legal_moves}
    assert all(entry[0] == key and entry[1] in legal and entry[2] > 0 and abs(entry[3]) <= 30000 for entry in entries)
    options = {"Book_File": str(output.resolve()), "OwnBook": "true", "Best_Book_Move": "true"}
    reader = Engine(args.reader, args.model, extra_options=options)
    try:
        move, _ = reader.go([], "nodes 64")
        validate_move(board, move)
        assert move in {legal[entry[1]] for entry in entries}, "USI did not select generated entry"
        base_value, base_policy = probe(reader)
    finally:
        reader.close()
    reader = Engine(args.reader, args.model, extra_options={
        "Book_File": str(output.resolve()), "Use_Book_Policy": "true"})
    try:
        blended_value, blended_policy = probe(reader)
        total = sum(entry[2] for entry in entries)
        counts = {legal[entry[1]]: entry[2] for entry in entries}
        for move, prior in base_policy.items():
            expected = .5 * (prior + counts.get(move, 0) / total)
            assert abs(blended_policy[move] - expected) < 1e-5, (move, blended_policy[move], expected)
        expected_value = .5 * (base_value + 1 / (1 + math.exp(-entries[0][3] * .0013226)))
        assert abs(blended_value - expected_value) < 1e-5
        cached_stats = reader.nn_stats()
        reader.send("setoption name MinibatchSize value 384")
        reader.send("isready")
        reader.until(lambda line: line == "readyok", 1800)
        ready_stats = reader.nn_stats()
        assert all(ready_stats[key] >= cached_stats[key] for key in cached_stats)
        repeated_value, repeated_policy = probe(reader)
        assert repeated_value == blended_value and repeated_policy == blended_policy
        assert reader.nn_stats() == ready_stats, "unchanged book must preserve blended NN cache"
        reader.send("setoption name Use_Book_Policy value false")
        reader.send("isready")
        reader.until(lambda line: line == "readyok", 1800)
        plain_value, plain_policy = probe(reader)
        assert abs(plain_value - base_value) < 1e-5 and plain_policy == base_policy
        assert reader.nn_stats()["evaluations"] > ready_stats["evaluations"], "book policy change must invalidate cache without resetting counters"
        # Editing the same book path must invalidate its cached policy too.
        changed = [list(entry) for entry in entries]
        changed[0][2] += 10
        changed[0][3] += 100
        output.write_bytes(b"".join(struct.pack("<QHHi", *entry) for entry in changed))
        reader.send("setoption name Use_Book_Policy value true")
        reader.send("isready")
        reader.until(lambda line: line == "readyok", 1800)
        probe(reader)  # Prime the modified book before a second edit while enabled.
        changed[0][2] += 10
        changed[0][3] += 100
        output.write_bytes(b"".join(struct.pack("<QHHi", *entry) for entry in changed))
        reader.send("isready")
        reader.until(lambda line: line == "readyok", 1800)
        changed_value, changed_policy = probe(reader)
        expected_changed_value = .5 * (base_value + 1 / (1 + math.exp(-changed[0][3] * .0013226)))
        assert abs(changed_value - expected_changed_value) < 1e-5
        changed_counts = {legal[entry[1]]: entry[2] for entry in changed}
        for move, prior in base_policy.items():
            expected = .5 * (prior + changed_counts.get(move, 0) / sum(entry[2] for entry in changed))
            assert abs(changed_policy[move] - expected) < 1e-5
        output.write_bytes(raw)  # Keep the original book for the MAKE_BOOK checks.
    finally:
        reader.close()
    writer = Engine(args.engine, args.model, extra_options={
        "Make_Book_Color": "black", "Use_Book_Policy": "true",
        "Use_Interruption": "false", "Save_Book_Interval": 1})
    try:
        blended_output = work / "blended.bin"
        if blended_output.exists():
            blended_output.unlink()
        writer.send(f"make_book {output.resolve()} {blended_output.resolve()} 128 1")
        writer.until(lambda line: line.startswith("merged entries\t"), 60)
        writer_value, writer_policy = probe(writer)
        assert abs(writer_value - expected_value) < 1e-5, "MAKE_BOOK warmup cache bypassed book value"
        for move, prior in base_policy.items():
            expected = .5 * (prior + counts.get(move, 0) / total)
            assert abs(writer_policy[move] - expected) < 1e-5, "MAKE_BOOK warmup cache bypassed book prior"
    finally:
        writer.close()
    args.report.write_text(json.dumps({"entries": len(entries), "generated": str(output),
                                      "readback": True, "policy_blend": True,
                                      "makebook_policy_blend": True, "engine_sha256": file_sha256(args.engine),
                                      "unchanged_book_cache_preserved": True, "policy_toggle_invalidates_cache": True,
                                      "same_path_book_edit_invalidates_cache": True,
                                      "reader_sha256": file_sha256(args.reader),
                                      "model_sha256": file_sha256(args.model),
                                      "book_sha256": file_sha256(output)}, indent=2) + "\n", encoding="utf-8")
    print("make_book generation, USI readback and policy blending passed", flush=True)


if __name__ == "__main__":
    main()
