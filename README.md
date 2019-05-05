# DeepLearningShogi

将棋でディープラーニングを実験するためのプロジェクトです。

基本的にAlphaGoの手法を参考に実装していく予定です。

検討経緯、実験結果などは、随時こちらのブログに掲載していきます。

http://tadaoyamaoka.hatenablog.com/

## 使用ライブラリ
* [Apery](https://github.com/HiraokaTakuya/apery)

※モンテカルロ木探索の実装は囲碁プログラムの[Ray+Rn](https://github.com/zakki/Ray)の実装を参考にしています。

## ソース構成
|フォルダ|説明|
|:---|:---|
|cppshogi|Aperyを流用した将棋ライブラリ（盤面管理、指し手生成）、入力特徴量作成|
|dlshogi|ニューラルネットワークの学習（Python）|
|make_hcp_by_self_play|Policyネットワークによる自己対局|
|make_hcpe_by_self_play|MCTSによる自己対局|
|test|テストコード|
|usi|対局用USIエンジン|
|utils|ツール類|

## ビルド環境
### USIエンジン、自己対局プログラム
#### Windowsの場合
* Windows 10 64bit
* Visual Studio 2017
#### Linuxの場合
* Ubuntu 16.04 LTS
* g++
#### Windows、Linux共通
* CUDA 10.0
* cuDNN 7.5

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* [Chainer](http://chainer.org/) 2以上
* Python 3.5以上 ([Anaconda](https://www.continuum.io/downloads))
* [Boost](http://www.boost.org/) 1.69.0

## ライセンス
ライセンスはMITライセンスとします。

cppshogiは[Apery](https://github.com/HiraokaTakuya/apery)のソースを流用しているため、GPLライセンスとします。
