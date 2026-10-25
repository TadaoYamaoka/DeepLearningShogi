"""Read migrated user ini files with archived legacy and current ONNX engines.

The configuration files are the staged user files; the small fixed NY network
is used for both engines to avoid building TensorRT plans for unrelated models.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, FIXTURES, ROOT, file_sha256


def parse(line):
    fields = line.split()
    return float(fields[4]), {move:float(value) for move,value in (field.split(":") for field in fields[5:])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location("migrate_ini", ROOT / "dlshogi/utils/migrate_classic_ini.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    # Preserve existing new values, BOM/line endings and legacy bytes exactly.
    original = b"\xef\xbb\xbfC_init=127\r\nFpuValue=700\r\n"
    converted = migration.migrate(original)
    assert converted.startswith(original) and converted.count(b"FpuValue=") == 1
    assert migration.migrate(converted) == converted
    assert b"CPuct=2540\r\n" in converted
    floated = b"C_init=127\r\nCPuct=2.54\r\nFpuValue=0.62\r\nPolicyTemperature=1.6\r\n"
    integers = migration.migrate(floated, from_float_options=True)
    assert integers.startswith(b"C_init=127\r\n")
    assert b"CPuct=2540\r\n" in integers and b"FpuValue=620\r\n" in integers
    assert b"PolicyTemperature=1600\r\n" in integers
    assert migration.migrate(integers) == integers
    try:
        migration.migrate(b"C_init=127\nC_init=128\n")
        raise AssertionError("duplicate ini key was accepted")
    except ValueError:
        pass
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"][0]
    work = args.report.parent / "ini_compat"
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for index, entry in enumerate(json.loads(args.manifest.read_text(encoding="utf-8"))):
        staged = Path(entry["staged"])
        assert file_sha256(staged) == entry["after_sha256"]
        model = work / f"fixture_{index}.onnx"
        shutil.copyfile(args.model, model)
        Path(str(model)+".ini").write_bytes(staged.read_bytes())
        old = Engine(args.old, model, extra_options={"DNN_Cache_Size":0})
        new = None
        try:
            _, lines = old.go(fixture["moves"], "nodes 1")
            legacy = [line for line in lines if line.startswith("info string nnprobe value ")]
            assert len(legacy) == 1
            new = Engine(args.new, model, extra_options={"DNN_Cache_Size":0})
            new.send("position startpos moves " + " ".join(fixture["moves"]))
            new.send("nnprobe")
            line = new.until(lambda line:line.startswith("info string nnprobe value "), 30)[-1]
            old_value, old_policy = parse(legacy[0])
            value, policy = parse(line)
            assert old_policy.keys() == policy.keys()
            error = max(abs(old_policy[move]-policy[move]) for move in policy)
            assert abs(old_value-value) < 1e-5 and error < 1e-5
            for key in ("C_init", "C_base", "C_fpu_reduction", "C_init_root", "C_base_root", "Softmax_Temperature"):
                assert any("Ignoring unknown model option: " + key == line.removeprefix("info string ") for line in new.transcript)
            assert any(line == "info string CPuct=2540" for line in new.transcript)
            assert any(line == "info string FpuValue=620" for line in new.transcript)
            temperature = "1.6" if "pre68" in staged.name else "1.4"
            assert "info string PolicyTemperature=" + str(int(float(temperature)*1000)) in new.transcript
            records.append(dict(ini_sha256=entry["after_sha256"], path=entry["path"],
                                temperature=float(temperature), max_policy_error=error,
                                value_error=abs(old_value-value), legacy_ready=True, current_ready=True))
        finally:
            old.close()
            (work / f"old_{index}.log").write_text("\n".join(old.transcript)+"\n",encoding="utf-8")
            if new:
                new.close()
                (work / f"new_{index}.log").write_text("\n".join(new.transcript)+"\n",encoding="utf-8")
    args.report.write_text(json.dumps(dict(complete=True, records=records,
        fixture_network_used_for_nn_compatibility=True, old_sha256=file_sha256(args.old),
        new_sha256=file_sha256(args.new), model_sha256=file_sha256(args.model)),indent=2)+"\n",encoding="utf-8")
    print("two migrated ini files accepted by legacy/current Release; NN value/policy agree",flush=True)


if __name__ == "__main__":
    main()
