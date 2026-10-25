"""FPU option migration, scalar units, model ini and quiescent reconfiguration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import cshogi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, FIXTURES, ROOT, file_sha256, validate_move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"][0]
    engine = Engine(args.engine, args.model, threads=1, slots=2)
    checks, searches = [], []
    try:
        assert not {"C_fpu_reduction", "C_fpu_reduction_root"} & engine.option_names
        declared = [line for line in engine.transcript if line.startswith("option name ")]
        for declaration in (
            "FpuStrategy type combo default reduction var reduction var absolute",
            "FpuStrategyAtRoot type combo default reduction var reduction var absolute var same",
            ):
            assert "option name " + declaration in declared
        for name, expected in dict(CPuct=2880, CPuctBase=28288, CPuctFactor=2000, CPuctAtRoot=2320, CPuctBaseAtRoot=25617, CPuctFactorAtRoot=2000, FpuValue=540, FpuValueAtRoot=0, PolicyTemperature=1740).items():
            line = next(line for line in declared if line.startswith(f"option name {name} type spin default "))
            assert int(line.split(" default ")[1].split()[0]) == expected
        assert not {"C_init", "C_base", "C_init_root", "C_base_root", "Softmax_Temperature"} & engine.option_names
        for name in ("CPuct", "CPuctBase", "CPuctFactor", "CPuctAtRoot", "CPuctBaseAtRoot", "CPuctFactorAtRoot", "RootHasOwnCpuctParams", "PolicyTemperature"):
            assert name in engine.option_names
        checks.append("lc0 FPU names/types advertised, retired names absent, integer scaled PUCT/FPU/temperature names advertised")
        engine.send("position startpos moves " + " ".join(fixture["moves"]))
        engine.send("nnprobe")
        original = engine.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1]
        configurations = (
            dict(FpuStrategy="reduction", FpuValue=540, FpuStrategyAtRoot="reduction", FpuValueAtRoot=0),
            dict(FpuStrategy="reduction", FpuValue=900, FpuStrategyAtRoot="same", FpuValueAtRoot=99000),
            dict(FpuStrategy="absolute", FpuValue=-1000, FpuStrategyAtRoot="same", FpuValueAtRoot=99000),
            dict(FpuStrategy="reduction", FpuValue=540, FpuStrategyAtRoot="absolute", FpuValueAtRoot=-1000),
        )
        for config in configurations:
            for key, value in config.items():
                engine.send(f"setoption name {key} value {value}")
            before = engine.nn_stats()
            engine.send("isready")
            engine.until(lambda line: line == "readyok", 1800)
            ready = engine.nn_stats()
            assert all(ready[key] >= before[key] for key in before)
            engine.send("position startpos moves " + " ".join(fixture["moves"]))
            engine.send("nnprobe")
            assert engine.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1] == original
            assert engine.nn_stats() == ready, "FPU-only changes discarded NN cache or altered NN evaluation"
            best, _ = engine.go(fixture["moves"], "nodes 512", timeout=120)
            validate_move(cshogi.Board(fixture["sfen"]), best)
            searches.append(dict(config=config, bestmove=best))
        checks.append("reduction 0.54/0.9, absolute -1 and root same/absolute search legally")
        checks.append("FPU-only isready preserves backend, cached scalar NN value and policy")
        for key, value in {"FpuStrategy":"invalid", "FpuValue":"nan",
                           "FpuStrategyAtRoot":"invalid", "FpuValueAtRoot":"100001"}.items():
            engine.send(f"setoption name {key} value {value}")
        engine.send("isready")
        engine.until(lambda line: line == "readyok", 1800)
        best, _ = engine.go(fixture["moves"], "nodes 128", timeout=120)
        validate_move(cshogi.Board(fixture["sfen"]), best)
        checks.append("invalid choices/nonfinite/out-of-range settings do not corrupt search")
    finally:
        engine.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.with_suffix(".usi.log").write_text("\n".join(engine.transcript)+"\n", encoding="utf-8")

    staging = args.report.parent / (args.report.stem + "_ini")
    staging.mkdir(parents=True, exist_ok=True)
    model = staging.resolve() / "parse-only.onnx"
    model.write_bytes(b"")  # No inference device enabled: test parsing without loading a new TensorRT plan.
    values = {"C_fpu_reduction":27, "C_fpu_reduction_root":5,
              "FpuStrategy":"absolute", "FpuValue":-1000,
              "FpuStrategyAtRoot":"same", "FpuValueAtRoot":250,
              "C_init":150, "C_base":30000, "C_init_root":120,
              "C_base_root":26000, "Softmax_Temperature":180,
              "CPuct":3000, "CPuctBase":30000, "CPuctFactor":2500,
              "CPuctAtRoot":2400, "CPuctBaseAtRoot":26000, "CPuctFactorAtRoot":2000,
              "RootHasOwnCpuctParams":"true", "PolicyTemperature":1800}
    Path(str(model)+".ini").write_text("".join(f"{key}={value}\n" for key,value in values.items()), encoding="utf-8")
    commands = ["usi", f"setoption name DNN_Model value {model}",
                "setoption name DNN_Slots value 0", "setoption name Mate_Root_Search value 0",
                "setoption name DfPn_Hash value 64", "isready", "quit"]
    env = os.environ.copy()
    env["PATH"] = r"C:\TensorRT-8.6.1.6\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;" + env.get("PATH", "")
    raw = subprocess.run([str(Path(args.engine).resolve())], input="\n".join(commands)+"\n", cwd=ROOT,
                         env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    (staging / "output.log").write_text(raw.stdout+raw.stderr, encoding="utf-8")
    assert raw.returncode == 1 and "At least one inference slot is required" in raw.stdout
    for key,value in values.items():
        expected = (f"info string Ignoring unknown model option: {key}" if key.startswith("C_") or key == "Softmax_Temperature"
                    else f"info string {key}={value}")
        assert expected in raw.stdout, (expected, raw.stdout[-2000:])
    checks.append("model ini warns/skips legacy search keys and reads integer/combo/bool keys")
    args.report.write_text(json.dumps(dict(complete=True, checks=checks, searches=searches,
        engine_sha256=file_sha256(args.engine), model_sha256=file_sha256(args.model)), indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
