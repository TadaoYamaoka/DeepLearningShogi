"""Exercise optimizer parameter types with mocked Optuna trials/matches; no GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace

source = (Path(__file__).resolve().parents[1] / "dlshogi/utils/usi_params_optimizer.py").read_text(encoding="utf-8")
tree = ast.parse(source)
defaults = {}
for node in ast.walk(tree):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument" and node.args
            and isinstance(node.args[0], ast.Constant)):
        for keyword in node.keywords:
            if keyword.arg == "default" and node.args[0].value in ("--init_params", "--suggest_params"):
                defaults[node.args[0].value] = ast.literal_eval(keyword.value)
objective = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "objective")
calls = []


def match(*args, **kwargs):
    assert type(args[2]["FpuValue"]) is int
    assert type(args[2]["CPuct"]) is int and type(args[2]["PolicyTemperature"]) is int
    kwargs["callback"](dict(engine1_won=1, draw=0, total=1))


namespace = dict(args=SimpleNamespace(
    suggest_params=defaults["--suggest_params"], init_params=defaults["--init_params"],
    command1="one", command2="two", name=None, games=2, byoyomi=1000, max_turn=320,
    opening=None, opening_moves=24, debug=False), options_list=[{}, {}],
    cshogi=SimpleNamespace(cli=SimpleNamespace(main=match)), optuna=SimpleNamespace(TrialPruned=Exception))
exec(compile(ast.Module(body=[objective], type_ignores=[]), "<optimizer objective>", "exec"), namespace)


class Trial:
    def __init__(self, number):
        self.number = number

    def suggest_float(self, key, low, high):
        calls.append((self.number, key, "float", low, high))
        return (low + high) / 2

    def suggest_int(self, key, low, high):
        calls.append((self.number, key, "int", low, high))
        return (low + high) // 2

    def report(self, *args):
        pass

    def should_prune(self):
        return False


for number in (0, 1):
    assert namespace["objective"](Trial(number)) == -1
assert any(key == "FpuValue" and kind == "int" and low == high == 540
           for number, key, kind, low, high in calls)
assert "C_fpu_reduction" not in source
print("optimizer integer FPU/PUCT/temperature parameters passed")
