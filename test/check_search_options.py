"""USI option migration, automatic workers and independent inference settings."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import cshogi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, ROOT, FIXTURES, file_sha256, validate_move

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))["positions"][0]
    engine = Engine(args.engine, args.model, threads=0, slots=2,
                    extra_options={"TaskWorkers":-1, "MinibatchSize":0,
                                   "DNN_Model2":"disabled-model-does-not-exist.onnx",
                                   "DNN_Slots2":0, "DNN_Batch_Size2":1})
    checks = []
    try:
        assert not any(name.startswith("UCT_Threads") for name in engine.option_names)
        for name in ("Threads", "TaskWorkers", "MinibatchSize", "MaxConcurrentSearchers", "DNN_Slots"):
            assert name in engine.option_names
        declared = [line for line in engine.transcript if line.startswith("option name ")]
        defaults = dict(Threads=0, TaskWorkers=-1, DNN_Slots=2, MinibatchSize=0, MaxConcurrentSearchers=1)
        for name,value in defaults.items():
            assert any(line.startswith(f"option name {name} type spin default {value} ") for line in declared)
        assert "option name Threads type spin default 0 min 0 max 128" in declared
        checks.append("new global/slot options advertised; UCT_Threads absent")
        checks.append("disabled GPU model ignored even when its batch is smaller")
        engine.send("position startpos moves " + " ".join(fixture["moves"]))
        engine.send("nnprobe")
        original_probe = engine.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1]
        original_stats = engine.nn_stats()
        assert original_stats["evaluations"] > 0
        for minibatch in (384, 64, 0, 0):
            engine.send(f"setoption name MinibatchSize value {minibatch}")
            engine.send("isready")
            engine.until(lambda line: line == "readyok", 1800)
            # isready's one-node warmup may prefetch additional positions.
            ready_stats = engine.nn_stats()
            assert all(ready_stats[key] >= original_stats[key] for key in original_stats)
            engine.send("position startpos moves " + " ".join(fixture["moves"]))
            engine.send("nnprobe")
            probe = engine.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1]
            assert probe == original_probe
            assert engine.nn_stats() == ready_stats, "isready discarded the previously cached evaluation"
            original_stats = ready_stats
        checks.append("minibatch-only changes and unchanged isready preserve backend counters and cached evaluation")
        engine.send("setoption name DNN_Cache_Size value 100001")
        engine.send("isready")
        engine.until(lambda line: line == "readyok", 1800)
        ready_stats = engine.nn_stats()
        assert all(ready_stats[key] >= original_stats[key] for key in original_stats)
        engine.send("position startpos moves " + " ".join(fixture["moves"]))
        engine.send("nnprobe")
        assert engine.until(lambda line: line.startswith("info string nnprobe value "), 30)[-1] == original_probe
        assert engine.nn_stats()["evaluations"] == ready_stats["evaluations"] + 1
        checks.append("cache capacity change invalidates cached evaluation without reloading the backend")
        for settings in (
                dict(Threads=0, DNN_Slots=2, TaskWorkers=-1, MinibatchSize=0),
                dict(Threads=1, DNN_Slots=2, TaskWorkers=0, MinibatchSize=384),
                dict(Threads=3, DNN_Slots=1, TaskWorkers=-1, MinibatchSize=64)):
            for name,value in settings.items():
                engine.send(f"setoption name {name} value {value}")
            engine.send("isready")
            engine.until(lambda line: line == "readyok", 1800)
            best,lines = engine.go(fixture["moves"], "nodes 1024", timeout=120)
            validate_move(cshogi.Board(fixture["sfen"]), best)
            assert engine.nn_stats() == engine.nn_stats()
        checks.append("automatic and explicit global workers independent of slots, repeated isready")
        checks.append("minibatches 0/64/384 including larger than GPU inference maximum")
    finally:
        engine.close()
    # A model .ini containing retired names must not insert a malformed option
    # into OptionsMap (which used to assert in Debug). No GPU is enabled here.
    staging = args.report.resolve().parent / (args.report.stem + "_ini")
    staging.mkdir(parents=True, exist_ok=True)
    model = staging / "model.onnx"
    shutil.copyfile(args.model, model)
    Path(str(model)+".ini").write_text("UCT_Threads=99\nThreads=1\n",encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = r"C:\TensorRT-8.6.1.6\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;"+env.get("PATH","")
    commands = ["usi", "setoption name DNN_Model value "+str(model),
                "setoption name DNN_Slots value 0", "setoption name Mate_Root_Search value 0",
                "setoption name DfPn_Hash value 64", "isready", "quit"]
    raw = subprocess.run([str(Path(args.engine).resolve())], input="\n".join(commands)+"\n",
                         cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    (staging / "output.log").write_text(raw.stdout+raw.stderr,encoding="utf-8")
    assert raw.returncode == 1, (raw.returncode,raw.stderr)
    assert "Ignoring unknown model option: UCT_Threads" in raw.stdout
    assert "info string Threads=1" in raw.stdout
    assert "At least one inference slot is required" in raw.stderr
    assert "info string engine error: At least one inference slot is required" in raw.stdout
    checks.append("retired model-ini setting safely ignored; known setting parsed; zero slots rejected")
    missing = staging / "missing-model.onnx"
    missing.unlink(missing_ok=True)
    commands = ["usi", "setoption name DNN_Model value " + str(missing),
                "setoption name DNN_Slots value 2", "setoption name Mate_Root_Search value 0",
                "setoption name DfPn_Hash value 64", "isready", "quit"]
    raw = subprocess.run([str(Path(args.engine).resolve())], input="\n".join(commands)+"\n",
                         cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    (staging / "missing_model.log").write_text(raw.stdout+raw.stderr,encoding="utf-8")
    assert raw.returncode == 1
    assert "info string engine error: Cannot open model for GPU 0: " + str(missing) in raw.stdout
    assert "engine error: Cannot open model for GPU 0:" in raw.stderr
    checks.append("missing active model reports fatal reason on USI stdout and stderr before exit")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(dict(complete=True, engine_sha256=file_sha256(args.engine),
        model_sha256=file_sha256(args.model), checks=checks, default_options=defaults, advertised_options=declared),indent=2)+"\n",encoding="utf-8")
    print(json.dumps(checks),flush=True)

if __name__ == "__main__":
    main()
