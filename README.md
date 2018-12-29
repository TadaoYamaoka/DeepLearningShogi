# DeepLearningShogi

将棋でディープラーニングを実験するためのプロジェクトです。

基本的にAlphaGoの手法を参考に実装していく予定です。

検討経緯、実験結果などは、随時こちらのブログに掲載していきます。

http://tadaoyamaoka.hatenablog.com/

## 使用ライブラリ
* [python-shogi](https://github.com/gunyarakun/python-shogi)
* [python-dlshogi](https://github.com/TadaoYamaoka/python-dlshogi)

## ビルド環境
### USIエンジン、自己対局プログラム
#### Windowsの場合
* Windows 10 64bit
* Visual Studio 2015
#### Linuxの場合 (Google Colab)
* Ubuntu 18.04 LTS
#### Windows、Linux共通
* CUDA 9.0
* cuDNN 7

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* Python 3.5もしくは3.6 ([Anaconda](https://www.continuum.io/downloads) 4.2.0 (64-bit))
* tensorflow-gpu
* keras

## ライセンス
ライセンスはMITライセンスとします。

# Windows 環境構築

## Anaconda 環境構築

``` dos
## For Python 3.6
conda create --name=DeepLearningShogi python=3.6
activate DeepLearningShogi
```

## GitHub リポジトリのクローン

``` dos
mkdir c:\work
cd c:\work
git clone https://github.com/TadaoYamaoka/DeepLearningShogi.git
cd DeepLearningShogi
```

## Git ブランチの切り替え

``` dos
git branch -a
git checkout remotes/origin/keras
```

## パッケージのインストール

``` dos
pip install tensorflow-gpu
pip install keras
pip install matplotlib
pip install python-shogi
python setup.py install
```

## 事前準備

* [floodgate] (https://ja.osdn.net/projects/shogi-server/releases/) の棋譜をダウンロード
* [7-Zip] (https://sevenzip.osdn.jp/) で解凍する

``` dos
mkdir data\floodgate
cd data\floodgate
wget http://iij.dl.osdn.jp/shogi-server/68500/wdoor2016.7z
7z x wdoor2016.7z

-> C:\work\DeepLearningShogi\data\floodgate\2016
```

* 棋譜リストのフィルター／ファイル準備

``` dos
cd C:/work/DeepLearningShogi
python ./utils/filter_csa_in_dir.py C:/work/DeepLearningShogi/data/floodgate/2016
python ./utils/prepare_kifu_list.py C:/work/DeepLearningShogi/data/floodgate/2016 kifulist
```

## 学習

``` dos
python train_policy_value_resnet.py --batchsize 32 --epoch 1 --log log kifulist_train.txt kifulist_test.txt

-> ./model/model_policy_value_resnet-best.hdf5
```

## 学習の継続

``` dos
python train_policy_value_resnet.py --batchsize 32 --epoch 1 --log log -m ./model/model_policy_value_resnet-best.hdf5 kifulist_train.txt kifulist_test.txt
```

## 将棋所で対局

* 将棋所を[ダウンロード] (http://shogidokoro.starfree.jp/download.html) して任意のディレクトリに解凍
* Shogidokoro.exe を起動
* 「対局」メニューの「エンジン管理」を選択
* 「C:\work\DeepLearningShogi\bat\parallel_mcts_player.bat」を「追加」
* 「閉じる」を選択
* 「対局」メニューの「対局」を選択
* 先手または後手の「エンジン」－「parallel_mcts_player」を選択し、「OK」を選択

