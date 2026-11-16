# Muon とパラメータ別 Optimizer の設定

モデルが公開するフックを使って、一部のパラメータを Muon などの Optimizer に割り当てられます。残りのパラメータは通常の Optimizer が更新します。タグの追加指定を省略すると、通常の Optimizer が全パラメータを担当します。

Muon を使用する場合は、`torch.optim.Muon` を提供する PyTorch と、対象を2次元パラメータとして保持するモデルが必要です。

## モデルのフック

フックは `model.named_parameters()` と一致するパラメータの完全名を返します。

- `<TAG>_parameters()`: タグに割り当てるパラメータ名。`matrix` タグなら `matrix_parameters()` を定義します。
- `no_weight_decay()`: weight decay を適用しないパラメータ名。省略可能で、通常側とタグ側の両方に適用されます。

たとえば、`hidden` と `head` という Linear 層を持つモデルでは、次のように中間層の重みだけを対象にできます。

```python
def matrix_parameters(self):
    return {"hidden.weight"}

def no_weight_decay(self):
    return {"hidden.bias", "head.bias"}
```

タグ名は任意です。Muon に割り当てる行列や weight decay の除外対象は、モデルの設計に合わせて明示します。パラメータの次元だけで自動的に振り分けることはありません。

## train.py

以下は `matrix_parameters()` を持つモデルでの設定例です。`exp__by10x192` は対応モデルの一例です。

```bash
python -m dlshogi.train TRAIN.hcpe3 TEST.hcpe \
  --network exp__by10x192 \
  --optimizer AdamW \
  --param_optimizer 'matrix=Muon(adjust_lr_fn="match_rms_adamw")' \
  --lr 0.0001 --weight_decay 0.01
```

`--lr` と `--weight_decay` は各 Optimizer のデフォルト値です。spec 内に `lr` や `weight_decay` を書いた場合は、その値を優先します。`--param_optimizer TAG=OPTIMIZER` は複数回指定でき、モデルの `<TAG>_parameters()` が返すパラメータ名を対象にします。`--weight_decay` に負数を指定した場合、各 Optimizer の指定値またはデフォルトを使用します。どの Optimizer でも `no_weight_decay()` の除外設定が適用されます。Optimizer と scheduler の指定は `()` を省略できます（例: `AdamW`、引数不要の `ConstantLR`）。scheduler の引数式では `optimizer` を参照できます。

## ptl.py

既存設定に追加設定を重ねます。モデルとデータのパスは環境に合わせて設定してください。付属の `config_muon.yaml` は対応モデルを含む設定例です。

```bash
python -m dlshogi.ptl fit \
  --config dlshogi/config.yaml --config dlshogi/config_muon.yaml
```

`param_optimizers.matrix` は `optimizer` と同じ `class_path` / `init_args` 形式です。通常側と Muon 側の学習率・weight decay はそれぞれ指定できます。設定例は両方を同じ値にしています。

[PyTorch Muon の公式ドキュメント](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)の `adjust_lr_fn="match_rms_adamw"` を明示し、AdamW の更新量に合わせる補正を使用します。

## scheduler・AMP・再開

タグを指定した場合、各 Optimizer は `MultiOptimizer` にまとめられます。scheduler はすべてのパラメータグループを更新し、AMP の unscale と勾配クリップは全パラメータに対して行います。異常な勾配を AMP が検出すると、その step の更新をすべての Optimizer でスキップします。Lightning は自動最適化を維持するため、勾配累積や global_step / EMA の周期も従来と同じです。

チェックポイントにはすべての Optimizer の状態と、タグ・クラス・group ごとのパラメータ名を保存します。同じ構成で保存・再開でき、タグの順序や対象パラメータが異なる場合は読み込みを拒否します。scheduler 未指定の `train.py` で再開する場合は、今回指定した各 group の学習率を使います。

タグが学習可能なパラメータを選ばない場合や、タグ間で対象が重複する場合はエラーになります。全パラメータをタグに割り当てた場合は default 側を省略します。`default` は予約語です。タグの指定がなければ通常の Optimizer を返します。

`MultiOptimizer` は closure なしで `step()` できる Optimizer が対象です。LBFGS など closure 必須のものは構築時に拒否します。`OneCycleLR` / `CyclicLR` では `cycle_momentum=False` が必要です。子は state を共有するため、保存・読み込みには親の `state_dict()` / `load_state_dict()` を使用してください。

## 実装・互換性

`dlshogi.optim.build_optimizer()` はフックに基づく振り分けを担当します。タグなしの場合の `_legacy_param_groups` は旧チェックポイントとの互換性のために空の decay group を保持します。タグありの場合の `_split_decay` は空の group を作りません。この違いを保つため、両者を分離しています。

推論用の融合や ONNX エクスポート、Attention の backend 選択は各モデルの実装に依存します。Optimizer の振り分けでは、モデルの forward や推論用の構造は変更しません。
