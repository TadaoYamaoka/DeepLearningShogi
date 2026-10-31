# DeepLearningShogi(dlshogi)
[![pypi](https://img.shields.io/pypi/v/dlshogi.svg)](https://pypi.python.org/pypi/dlshogi)

将棋でディープラーニングの実験をするためのプロジェクトです。

基本的にAlphaGo/AlphaZeroの手法を参考に実装していく方針です。

検討経緯、実験結果などは、随時こちらのブログに掲載していきます。

http://tadaoyamaoka.hatenablog.com/

## ダウンロード
[Releases](https://github.com/TadaoYamaoka/DeepLearningShogi/releases)からダウンロードできます。

最新のモデルファイルは、[棋神アナリティクス](https://kishin-analytics.heroz.jp/lp/)でご利用いただけます。

## ソース構成
|フォルダ|説明|
|:---|:---|
|cppshogi|Aperyを流用した将棋ライブラリ（盤面管理、指し手生成）、入力特徴量作成|
|dlshogi|ニューラルネットワークの学習（Python）|
|dlshogi/utils|ツール類|
|selfplay|MCTSによる自己対局|
|test|テストコード|
|usi|対局用USIエンジン|
|usi_onnxruntime|OnnxRuntime版ビルド用プロジェクト|

## ビルド環境
### USIエンジン、自己対局プログラム
#### Windowsの場合
* Windows 11 64bit
* Visual Studio 2022
#### Linuxの場合
* Ubuntu 18.04 LTS / 20.04 LTS
* g++
#### Windows、Linux共通
* CUDA 12.1
* cuDNN 8.9
* TensorRT 8.6

TensorRT 8をデフォルトとし、`TENSORRT10`を定義するとTensorRT 10向けにビルドします。
Visual StudioのTensorRT使用プロジェクト（x64）は、`TENSORRT10`定義、TensorRT 10.16.1、CUDA 13.4の設定です。
TensorRT 8に戻す場合は、定義を外し、インクルード・ライブラリ・CUDAのパスと
リンク対象（`nvinfer.lib;nvparsers.lib;nvonnxparser.lib`）をTensorRT 8環境に合わせてください。
Linuxでは`make TENSORRT10=1`で切り替えます（TensorRT 10のヘッダー・ライブラリへのパスは環境に合わせて設定）。
TensorRT 10のエンジンキャッシュはファイル名に`.trt10`を付け、TensorRT 8と区別します。
推論には対応GPUでFP16、それ以外ではFP32を使用します。

※CUDA 10.0以上であれば変更可

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* [Pytorch](https://pytorch.org/) 1.6以上
* Python 3.7以上 ([Anaconda](https://www.continuum.io/downloads))
* CUDA (PyTorchに対応したバージョン)
* cuDNN (CUDAに対応したバージョン)

#### Weight Decayからの除外

既定ではモデルの除外指定を使わず、全パラメータにWeight Decayを適用します。
通常の学習CLIでは`--use_no_weight_decay`、Lightningでは
`--model.use_no_weight_decay=true`（YAMLでは`model`配下に`use_no_weight_decay: true`）で
モデルの除外指定を有効にできます。モデルの`no_weight_decay()`で
`named_parameters()`の完全なパラメータ名の集合を返して指定します。
例えば`{"absolute_pos_embed", "blocks.8.relative_position_bias_table"}`のように返します。
フックのないモデルや除外対象のないモデルは、従来どおり1グループのままです。
optimizerの種類やWeight Decayの係数は既存の設定を使います。

`policy_value_network_exp__bm_fuse`では`absolute_pos_embed`と
`relative_position_bias_table`だけが除外され、BatchNormや通常のbiasにはDecayを適用します。
1次元のパラメータも、モデルが明示しなければDecayの対象です。

除外の有無や除外対象を切り替えるとoptimizerのグループ構成が変わるため、旧optimizer状態は
そのまま読み込めません。通常のCLIで学習checkpointから切り替える場合は
`--resume checkpoint.pth --reset_optimizer --reset_scheduler --use_no_weight_decay`を
指定し、モデルの重みを引き継いでoptimizer・schedulerを新規開始します
（epochとstepは再開元を引き継ぎます）。Lightningでは`--ckpt_path`による完全再開を使わず、
`--model.resume_model checkpoint.pth --model.use_no_weight_decay=true`で
通常CLIのcheckpoint（`model`キーの重み）を読み込んで新規開始します。
Lightningの`.ckpt`から切り替える場合は、以下のように学習モデルの重みだけを
抽出し、出力した`weights.pth`を`resume_model`へ渡してください。

```python
import torch

checkpoint = torch.load("old.ckpt", map_location="cpu", weights_only=False)
weights = {
    name[len("model."):]: value
    for name, value in checkpoint["state_dict"].items()
    if name.startswith("model.")
}
torch.save({"model": weights}, "weights.pth")
```

同じ除外設定で保存したcheckpointは従来どおり完全再開できます。

## 謝辞
* 将棋の局面管理、合法手生成に、[Apery](https://github.com/HiraokaTakuya/apery)のソースコードを使用しています。
* モンテカルロ木探索の実装は囲碁プログラムの[Ray+Rn](https://github.com/zakki/Ray)の実装を参考にしています。
* 探索部の一部にLeela Chess Zeroのソースコードを流用しています。
* 王手生成などに、[やねうら王](https://github.com/yaneurao/YaneuraOu)のソースコードを流用しています。

## ライセンス
ライセンスはGPL3ライセンスとします。
