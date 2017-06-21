# DeepLearningShogi

将棋でディープラーニングを実験するためのプロジェクトです。

基本的にAlphaGoの手法を参考に実装していく予定です。

検討経緯、実験結果などは、随時こちらのブログに掲載していきます。

http://tadaoyamaoka.hatenablog.com/

## 使用ライブラリ
* [Chainer](http://chainer.org/) 2.0.0
* [elmo_for_learn](https://github.com/mk-takizawa/elmo_for_learn)
* [Ray+Rn](https://github.com/zakki/Ray)

※2017/06/02 高速化のため [python-shogi](https://github.com/gunyarakun/python-shogi)  の使用はやめました。

## ビルド環境
* Windows 10 64bit
* Python 3.5 ([Anaconda](https://www.continuum.io/downloads) 4.2.0 (64-bit))
* Visual Studio 2015
* [Boost](http://www.boost.org/) 1.64

## ライセンス
ライセンスはMITライセンスとします。

cppshogiは[elmo_for_learn](https://github.com/mk-takizawa/elmo_for_learn)のソースを流用しているため、GPLライセンスとします。
