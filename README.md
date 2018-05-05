# DeepLearningShogi

将棋でディープラーニングを実験するためのプロジェクトです。

基本的にAlphaGoの手法を参考に実装していく予定です。

検討経緯、実験結果などは、随時こちらのブログに掲載していきます。

http://tadaoyamaoka.hatenablog.com/

## 使用ライブラリ
* [elmo_for_learn](https://github.com/mk-takizawa/elmo_for_learn)

※モンテカルロ木探索の実装は囲碁プログラムの[Ray+Rn](https://github.com/zakki/Ray)の実装を参考にしています。

## ビルド環境
### USIエンジン、自己対局プログラム
#### Windowsの場合
* Windows 10 64bit
* Visual Studio 2015
#### Linuxの場合
* Ubuntu 16.04 LTS
* g++
#### Windows、Linux共通
* CUDA 9.0
* cuDNN 7

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* [Chainer](http://chainer.org/) 2以上
* Python 3.5もしくは3.6 ([Anaconda](https://www.continuum.io/downloads) 4.2.0 (64-bit)※Chainerが4系の場合はAnadonda5系が必要)
* [Boost](http://www.boost.org/) 1.65.1

## ライセンス
ライセンスはMITライセンスとします。

cppshogiは[elmo_for_learn](https://github.com/mk-takizawa/elmo_for_learn)のソースを流用しているため、GPLライセンスとします。
