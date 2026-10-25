"""Read actual session values after USI setoption/isready; check subtree reuse."""
import argparse
import json
import math
from pathlib import Path
import sys
import cshogi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_classic import Engine, file_sha256, validate_move


def read_settings(engine):
    engine.send("searchsettings")
    prefix = "info string searchsettings "
    line = engine.until(lambda line: line.startswith(prefix), 30)[-1]
    return json.loads(line[len(prefix):])


def set_ready(engine, settings):
    for key, value in settings.items():
        engine.send(f"setoption name {key} value {value}")
    engine.send("isready")
    engine.until(lambda line: line == "readyok", 1800)


def require(actual, expected):
    for key, value in expected.items():
        if type(value) is float:
            assert math.isclose(actual[key], value, abs_tol=1e-6), (key, actual[key], value)
        else:
            assert actual[key] == value, (key, actual[key], value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    checks, reuse_results = [], []
    engine = Engine(args.engine, args.model, threads=1, slots=1)
    try:
        # Every one of the 12 public lc0-named options is set through USI,
        # without a model ini; adjacent dlshogi session controls are checked too.
        supplied = dict(CPuct=3250, CPuctBase=31000, CPuctFactor=4500,
            CPuctAtRoot=7000, CPuctBaseAtRoot=35000, CPuctFactorAtRoot=5000,
            RootHasOwnCpuctParams="false", FpuStrategy="absolute", FpuValue=-1000,
            FpuStrategyAtRoot="same", FpuValueAtRoot=1000, PolicyTemperature=1750,
            Threads=1, MinibatchSize=64, TaskWorkers=1, MaxConcurrentSearchers=0,
            ReuseSubtree="false", MultiPV=2, PV_Interval=77, Eval_Coef=900,
            Resign_Threshold=111, Draw_Ply=300, Draw_Value_Black=411, Draw_Value_White=577,
            DNN_Cache_Size=100003, DNN_Model2="unused-model.onnx", DNN_Batch_Size2=16,
            DebugMessage="true", Const_Playout=17, USI_Ponder="true", Stochastic_Ponder="false",
            Random_Ply=17, Random_Temperature=8000, Random_Temperature_Drop=100,
            Random_Cutoff=20, Random_Cutoff_Drop=5, Random2_Ply=19,
            Random2_Probability=60, Random2_Temperature=9000, Random2_Cutoff=40,
            Random2_Value_Limit=800)
        set_ready(engine, supplied)
        actual = read_settings(engine)
        expected = dict(CPuct=3.25, CPuctBase=31000.0, CPuctFactor=4.5,
            CPuctAtRoot=7.0, CPuctBaseAtRoot=35000.0, CPuctFactorAtRoot=5.0,
            RootHasOwnCpuctParams=False, FpuStrategy="absolute", FpuValue=-1.0,
            FpuStrategyAtRoot="same", FpuValueAtRoot=1.0, PolicyTemperature=1.75,
            ClassicPolicyTemperature=1.0, EffectiveCPuctAtRoot=3.25, EffectiveFpuValueAtRoot=-1.0,
            Threads=1, MinibatchSize=64, TaskWorkers=1, MaxConcurrentSearchers=0,
            ReuseSubtree=False, MultiPV=2, PV_Interval=77, Eval_Coef=900,
            Resign_Threshold=.111, Draw_Ply=300, Draw_Value_Black=.411, Draw_Value_White=.577,
            DNN_Cache_Size=100003, DNN_Model2="unused-model.onnx", DNN_Batch_Size2=16,
            DebugMessage=True, Const_Playout=17, PonderingMode=True,
            Random_Ply=17, Random_Temperature=8.0, Random_Temperature_Drop=.1,
            Random_Cutoff=.02, Random_Cutoff_Drop=.005, Random2_Ply=19,
            Random2_Probability=.06, Random2_Temperature=9.0, Random2_Cutoff=.04,
            Random2_Value_Limit=.8, DNN_Slots=1, DNN_Batch_Size=128)
        require(actual, expected)
        checks.append("all twelve lc0-named options and adjacent session controls reflected from USI")
        engine.send("setoption name CPuct value 100")
        require(read_settings(engine), {"CPuct":3.25})
        set_ready(engine, dict(RootHasOwnCpuctParams="true", FpuStrategyAtRoot="absolute", FpuValueAtRoot=-500))
        after = read_settings(engine)
        require(after, dict(CPuct=.1, EffectiveCPuctAtRoot=7.0, EffectiveFpuValueAtRoot=-.5))
        checks.append("ready snapshot updates only after isready; root-specific settings take effect")
        for key, value, warning in (("CPuct","2.5","1/1000"),
                ("FpuValue","nan","1/1000"), ("CPuctBase","0","outside"),
                ("FpuStrategy","invalid","invalid value or choice")):
            engine.send(f"setoption name {key} value {value}")
            line = engine.until(lambda line:line.startswith("info string Rejected option " + key + ":"),30)[-1]
            assert warning in line
        set_ready(engine, {})
        require(read_settings(engine), dict(CPuct=.1, FpuValue=-1.0, CPuctBase=31000.0, FpuStrategy="absolute"))
        checks.append("rejected fractional/range/choice values explain why and preserve previous values")
    finally:
        engine.close()
        args.report.with_suffix(".usi.log").write_text("\n".join(engine.transcript)+"\n",encoding="utf-8")

    # Inspect visits at the real root immediately before each Search is built.
    # This avoids relying on stopper timing/NN prefetch differences in info nodes.
    for reuse in (True, False):
        engine = Engine(args.engine, args.model, threads=1, slots=1,
                        extra_options={"ReuseSubtree":str(reuse).lower(), "DNN_Cache_Size":0})
        try:
            best, _ = engine.go([], "nodes 3000", timeout=180)
            board = cshogi.Board()
            validate_move(board, best)
            board.push_usi(best)
            reply, lines = engine.go([best], "nodes 1", timeout=120)
            validate_move(board, reply)
            settings = read_settings(engine)
            assert settings["ReuseSubtree"] is reuse
            before = settings["LastRootVisitsBeforeSearch"]
            assert (before > 0) if reuse else (before == 0), (reuse, settings)
            nodes = [int(line.split(" nodes ",1)[1].split()[0]) for line in lines if " nodes " in line]
            reuse_results.append(dict(reuse=reuse, retained_visits=before, final_nodes=nodes[-1], bestmove=best, reply=reply))
        finally:
            engine.close()
            (args.report.parent / (args.report.stem+f"_reuse_{reuse}.log")).write_text("\n".join(engine.transcript)+"\n",encoding="utf-8")
    checks.append("ReuseSubtree true retains child visits; false starts with zero visits after advancing root")
    args.report.write_text(json.dumps(dict(complete=True, checks=checks, supplied=supplied,
        actual=actual, expected=expected, reuse=reuse_results,
        engine_sha256=file_sha256(args.engine), model_sha256=file_sha256(args.model)),indent=2)+"\n",encoding="utf-8")
    print(json.dumps(checks),flush=True)


if __name__ == "__main__":
    main()
