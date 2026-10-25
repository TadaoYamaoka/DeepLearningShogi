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

※CUDA 10.0以上であれば変更可

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* [Pytorch](https://pytorch.org/) 1.6以上
* Python 3.7以上 ([Anaconda](https://www.continuum.io/downloads))
* CUDA (PyTorchに対応したバージョン)
* cuDNN (CUDAに対応したバージョン)

## 謝辞
* 将棋の局面管理、合法手生成に、[Apery](https://github.com/HiraokaTakuya/apery)のソースコードを使用しています。
* モンテカルロ木探索の実装は囲碁プログラムの[Ray+Rn](https://github.com/zakki/Ray)の実装を参考にしています。
* USIの探索部にLeela Chess Zeroのclassic探索を使用しています。
* 王手生成などに、[やねうら王](https://github.com/yaneurao/YaneuraOu)のソースコードを流用しています。

## ライセンス
ライセンスはGPL3ライセンスとします。

## lc0 classic 探索

USI の探索は `lc0/` の classic を使用します。upstream の基準コミットは
同ディレクトリの `UPSTREAM_REVISION` に記録しています。将棋のルール、USI、モデル、
特徴量、方策ラベル、TensorRT / ONNX Runtime の推論は dlshogi / cppshogi のものです。
`lc0/` は必要なupstreamソースを保持するビルド対象、`lc0_shogi/` は将棋との接続層です。
比較用の完全なupstreamチェックアウトは、リポジトリ外の `../lc0/` に置きます。
差分検査スクリプトは、このチェックアウトのHEADではなく `UPSTREAM_REVISION` のGitオブジェクトを参照します。
`usi/` 以下にはlc0のソースや接続層を置きません。

Windows の USI ビルドには Visual Studio 2022 / C++20 を使用します。
`NyugyokuFeatures` でモデルの入玉特徴量の有無を指定してください。TensorRT 版の既定は
`true`、ONNX Runtime 版と multiponder の既定は `false` です。
異なる構成を並行して使う場合は、出力先と中間ディレクトリを分けます。

```powershell
MSBuild.exe usi\usi.vcxproj /p:Configuration=Release /p:Platform=x64 /p:NyugyokuFeatures=true
MSBuild.exe usi_onnxruntime\usi_onnxruntime.vcxproj /t:Restore /p:RestorePackagesConfig=true
MSBuild.exe usi_onnxruntime\usi_onnxruntime.vcxproj /p:Configuration=Release /p:Platform=x64 /p:NyugyokuFeatures=false
MSBuild.exe test\classic_tests.vcxproj /p:Configuration=Release /p:Platform=x64
usi\x64\Release\classic_tests\classic_tests.exe
```

ONNX Runtime 版は opset 14 のモデルに対応するため、依存を ONNX Runtime 1.20.1 /
DirectML 1.15.2 に更新しています。モデルの入力形状とビルド時の特徴量が合わない場合は
初期化時にエラーにします。モデルと生成される TensorRT キャッシュの保存先には書き込み権限が必要です。

NN は勝率を scalar Q に変換し、NN の WDL 推定は使いません。ルール上の引き分けは
classic の標準フィールドで保持します。`Draw_Value_Black` / `Draw_Value_White` が
互いに補数でない場合、各色が常に自分の設定値で引き分けを評価します。この点は、
引き分けに到達した側によって祖先の値が変わっていた旧探索と異なります。
USI の `info` には `wdl` を出力しません。

selfplay は独自の探索を維持し、旧 Node を `selfplay/` 内に専用化しています。
selfplay を classic に移す作業は、複数対局のバッチ推論性能を検証する別フェーズです。

### 探索の回帰試験

Python の対局裁定には `cshogi==1.1.0` を使用します。固定100局面は
`test/classic_positions.json` に保存しており、比較中に生成し直しません。
`test/build_legacy_probe.py` は固定コミットの旧コードを `artifacts/` 内に復元し、
NN 出力を観測する比較用実行形式を生成します。製品には旧探索をリンクしません。

```powershell
python -m pip install cshogi==1.1.0
python test\build_legacy_probe.py --nyugyoku
python test\check_usi_protocol.py --engine ENGINE.exe --model MODEL.onnx
python test\evaluate_classic.py compare --old OLD.exe --new NEW.exe --model MODEL.onnx
python test\evaluate_classic.py match --old OLD.exe --new NEW.exe --model MODEL.onnx
```

固定回数比較は１スレッド・800回、ponder・定跡・ランダム着手・ルート dfpn 無効で行います。
最善手一致率50%未満、または先後交換40局の得点率20%未満なら、原因を調査して判断します。
これらは重大な実装異常の検出基準で、Elo による棋力評価ではありません。
NN の直接比較には、USI 拡張の `nnprobe` と `evaluate_classic.py nncompare` を使用します。

lc0 の著作権表示と `COPYING` を保持しています。GPL 系ソースの統合と、TensorRT 等を
リンクしたバイナリの配布条件は区別し、配布時には追加許諾の適用範囲を別途確認してください。

### 実装の境界

|対象|扱い|
|:--|:--|
|旧 USI `UctSearch.*` / `Node.*` / `Message.*` / `PvMateSearch.*`|削除。自己対局用 Node は `selfplay/Node.*` に専用化|
|`lc0/src/search/classic/`|探索・選択・バックアップ・木管理を使用。classic の wrapper と既成時間配分器はリンクしない|
|`lc0/src/utils/`|mutex、計時、オプション、ログ、乱数などを明示リストから使用|
|lc0 chess / UCI / Syzygy / protobuf / NN backend / caching wrapper|リンクしない。`lc0_shogi/compat/` と dlshogi 側の実装で境界を置き換える|
|`dag_classic` など他の探索|取り込まない|
|cppshogi / `dfpn.*` / `mate.h` / `nn_tensorrt.*` / ONNX 推論|維持。モデル形状検査と ONNX 入力要素数の修正を追加|

upstream 本体の変更は6ファイルです。変更理由は
[`lc0/CHANGES.txt`](lc0/CHANGES.txt) に列挙しています。
`lc0_shogi/lc0_shogi.vcxproj` と `lc0_shogi/classic.mk` はソースを列挙し、Visual Studio のビルド前には
`test/check_classic_sources.ps1` が旧配置の残存と、チェス実装、別探索、旧探索、別 lc0 コピーの
混入を拒否します。互換ヘッダの優先順位と、コンパイル対象の限定を組み合わせています。

将棋の指し手は Edge 内では16ビットの `proFromAndTo()` を保持し、必要時に
`move16toMove()` で局面から捕獲駒などを復元します。NN 方策インデックスには lc0 の
チェス用インデックスを使わず、`make_move_label(move, color)` を使います。
バックエンドの合法手順と探索が要求する順序は指し手IDで対応付けます。
Edge は4バイト、Node は64バイトです。32ビット Move をそのまま格納した8バイト Edge
に比べ、562合法手の試験局面では Edge 配列が4,496バイトから2,248バイトになります。
実際の全メモリ量には子ノード、アロケータ、NNキャッシュなどが加わります。

各 worker は独立した cppshogi `Position` とアドレスの動かない `StateInfo` スタックを
所有し、Append/Pop は `doMove`/`undoMove` です。履歴のコピーは独立カーソルの初期化時に
一度再生し、以後の代入は共通接頭辞まで undo して差分だけ進めます。
局面の shallow copy に `StateInfo` の寿命を預けません。
ルートを進めても全履歴と絶対手数を保持し、巻き戻しや別枝で木を捨てる場合は
PV dfpn の探索済み記録も無効にします。

### 評価・ルール・詰みの反映

NN の勝率 `p` は `q=2p-1, d=0, m=0` として渡します。WDL rescale、contempt、moves-left、
Dirichlet noise は無効です。方策温度`PolicyTemperature`とキャッシュはdlshogi側で一度だけ適用します。
classic内部の方策温度は1.0に固定します。USIの数値設定は整数`spin`とし、
PUCT・FPU・方策温度は1000で割ってlc0の単位へ戻します。`CPuctBase*`は整数のまま渡します。

ルールによる引き分けだけを `wl=0,d=1` で保持します。`GetQ(draw_score)` には、その手を
選ぶ色の `2*Draw_Value/1000-1` を渡すため、400/400 の非補数設定も表せます。
NN の引き分け推定を合成しません。USI の評価値は `Eval_Coef*log(p/(1-p))`、端点は
±30000です。`score mate` の距離は合法な詰みPVで確認できるものを出力します。
引き分け値0/1000も勝敗確定と混同せず、最大手数や千日手を越えた詰みPVは表示しません。
未確定の NPS は負数を出さずフィールドを省略します。

リーフでは旧探索と同様に、最大手数、`isDraw(16)` の千日手・連続王手・優劣判定、
短手数詰み、入玉、合法手なしを cppshogi で判定します。チェスの50手ルール、
ステイルメイト、駒不足の判定は実行しません。ルートの入玉宣言と定跡は既存 USI 層が扱います。

|詰み探索|実行場所と反映|
|:--|:--|
|ルート dfpn|`main.cpp` の既存の並列スレッド。発見すると探索を止め、dfpn の手と詰みPVを採用。`Mate_Root_Search` / `DfPn_Min_Search_Millisecs` を維持|
|短手数詰み|classic の ExtendNode から `lc0_shogi/rules.cpp` を呼び、NN投入前に terminal / bounds を設定。既定5手、Makefileの `MATE_SEARCH_DEPTH` を維持|
|PV dfpn|`lc0_shogi/mate_coordinator.*` の専用プール。`PV_Mate_Search_Threads/Depth/Nodes` を維持。PVを指し手列で再解決し、`ApplyProof()` が反映|

PV の反映は `nodes_mutex_` の排他ロック下で、訪問済みの対象を terminal にし、祖先の
bounds を即時更新します。経路にも対象にも in-flight=0 を要求しません。未訪問の対象は
再試行し、初回展開中の兄弟の bounds は未確定として扱います。過去訪問の寄与補正や
架空の訪問加算はしません。保留中の NN 計算が戻っても、classic の
`DoBackupUpdateSingleNode()` が途中の terminal 値を使ってバックアップします。
`MakeSolid()` による移動に備え、dfpn 側に Node の生ポインタは保持しません。

### USI とビルドの互換性

UCI エンジンループは使わず、既存の `main.cpp` から `lc0_shogi/search_session.*` を呼びます。
時間配分、時間延長、ノード上限、ponder中フラグは独自 SearchStopper が扱います。
ponderhit のための classic API は追加していません。セッションの応答ゲートが、
勝敗確定による内部停止・入玉・定跡・ルートdfpnの早期完了も含め、stop/ponderhit まで
bestmove を保留します。応答スレッドと補助エンジンは終了時に join します。

`DNN_Model*`、`DNN_Batch_Size*`、キャッシュ、MultiPV、木再利用、
引き分け値、投了、ランダム着手、定跡と定跡方策、make_book、multiponder の設定名を維持します。
探索の並列設定は`Threads`・`TaskWorkers`・`MinibatchSize`、推論の並列度は`DNN_Slots*`へ分離し、
`UCT_Threads*`は廃止しました。末尾の移行表を参照してください。
モデルや GPU 設定を変更した場合は isready で再設定します。探索の選択式は classic なので、
同じオプション値が旧探索と全く同じ手順・訪問分布を保証するものではありません。

Visual Studio プロジェクトは静的な `lc0_shogi` を参照し、元の Meson への移行は不要です。
`lc0_shogi/` 内のコンパイルの分担は次のとおりです。推論マクロに依存する実装は各実行形式でコンパイルします。

| ビルド対象 | コンパイルするソース |
|:--|:--|
| `lc0_shogi.vcxproj`（静的ライブラリ） | 明示した `../lc0/src/` のclassic・utils・shared_paramsと、`position_adapter.cpp`、`rules.cpp`、`backend_support.cpp`、`classic_options.cpp` |
| USI / ONNX Runtime / multiponder（各実行形式） | `backend.cpp`、`search_session.cpp`、`mate_coordinator.cpp` |

`compat/` のincludeパスは、各構成とも `lc0/src/` より先に指定します。
配置移行後の確認結果は `test/layout_review.json` に記録しています。
10構成のWindowsビルド、Release / Debugのユニット試験、TensorRT / ONNX Runtimeの
NYあり・なし4構成での設定反映・木再利用試験を通過しました。
移動前との固定30局面×2モデル・4,000ノード比較では、最善手・PV・NN評価件数・
バッチ数がすべて一致しました。訪問数のみ、停止監視のタイミングによる差が60局面中4局面にあります。
移動した実装の内容はバイト単位で同一で、lc0本体の改変は従来の6ファイルのままです。
Makefile も同じ明示リストを使います。GNU make の make_book は通常版と別の
オブジェクトディレクトリを使い、構成マクロの異なる生成物を混ぜません。
Linux には C++20 の標準ライブラリを備えたコンパイラが必要です。

追加の回帰試験:

```powershell
python test\build_legacy_probe.py --onnx --nyugyoku
python test\check_makebook.py --engine MAKE_BOOK.exe --reader ENGINE.exe --model MODEL.onnx
python test\check_multiponder.py --engine MULTI.exe --auxiliary ENGINE.exe --model MODEL.onnx
python test\evaluate_classic.py benchmark --old OLD.exe --new NEW.exe --model MODEL.onnx --threads 2 --nodes 20000 --positions 30 --repeats 3 --pv-threads 0
# PV dfpn 有効時は同じ条件で --pv-threads 2 を指定
```

NN の1訪問試験ではキャッシュを無効にして訪問数1を検査し、ルート方策上位3手と評価値を
旧NN出力と比較します。キャッシュを有効にした classic は、停止判定前にキャッシュヒットを
複数処理できるため、`go nodes 1` だけでは1訪問を保証しません。
NN の厳密比較は同一バックエンド・同一モデルで行います。TensorRT の半精度推論と
ONNX Runtime の比較に、そのまま同一の許容誤差を適用しません。
速度測定は他のGPUジョブがない状態で実行し、棋力・速度の合格閾値とは分離します。
詰み確定で指定ノード数より早く停止した局面は、速度の中央値から除外して別に記録します。
`test/collect_classic_acceptance.py` は完了済みの各レポートと実行形式のハッシュを検査し、
`test/classic_acceptance.json` に受け入れ試験の要約を出力します。

### NPS レビュー後の変更

推論スロットは `CreateComputation()` ごとの巡回割り当てを廃止し、
`ComputeBlocking()` で空きスロットを借り、完了・例外時に返します。
バッチ収集中にスロットを保持しません。TensorRT の入力・方策・価値バッファは
スロットごとに `cudaHostAlloc` で一度確保して再利用し、GPU の同期完了後に読みます。
ONNX Runtime もスロットごとのバッファを再利用しますが、CUDA に依存しません。
これにより、バッチごとの大きな出力配列の確保・ゼロ初期化もなくなります。

PV dfpn の未訪問対象への再試行は、プール全体で前回の再試行後にバックアップ数が
増えた場合に行います。探索終了時の最終再試行は維持します。履歴識別文字列は
探索開始時に一度作り、ルート移動時に新ルートの子孫以外の探索済み記録を除去します。
`ApplyProof()` から、別の mutex に属する最終 bestmove/ponder 値への書き込みを削除しました。

classic 本体のバッチ収集、衝突許容量、`MaxPrefetch=32`、`IdlingMinimumWork=0` は
変更していません。短い探索の平均バッチサイズによる速度差は、バッファ修正とは別に評価します。
ベンチマークは同じ固定局面集合から選び、反復ごとに新旧の実行順を入れ替えます。
レポートには実行形式・モデルの SHA256、実ノード数、各局面・反復の所要時間、
反復別の速度比、早期終局による除外を保存します。

今回の修正後の試験・速度測定は `test/classic_performance.json` に別途記録します。
`python test/collect_classic_performance.py` で、4構成のNN比較・2スレッドUSI試験と
両モデル×PV dfpn無効/2スレッドの速度レポートの実行形式ハッシュを照合して集計します。
`classic_acceptance.json` の40局対局結果は以前の実行形式に対する証跡として保持します。

RTX 4090 / TensorRT 8.6.1.6、探索2スレッド、20,000ノード、batch 128、
固定30局面×3回の測定結果（速度比は修正版/旧探索）:

|モデル|PV dfpnスレッド|旧NPS中央値|修正版NPS中央値|NPS中央値の比|局面ごとの比の中央値|
|:--|--:|--:|--:|--:|--:|
|NY|0|11,015|11,866|1.077|1.092|
|NY|2|11,686|11,724|1.003|1.029|
|no-NY|0|20,360|20,244|0.994|1.033|
|no-NY|2|21,543|21,160|0.982|0.952|

全360組が指定ノード数に到達し、早期終局による除外はありませんでした。
各条件は新旧それぞれ同じプロセスで3回反復し、木の再利用は無効、NNキャッシュは保持します。
初期化・ウォームアップを計時から除き、新旧は交互に実行しました。
これは旧探索と修正版の比較であり、個々の修正の改善率を分離した測定ではありません。
1スレッド・4,000ノードで報告された約91%との直接比較には使えません。

PV dfpn 有効時のNPSには、新旧の終端ノード再訪問の数え方の違いが混ざります。
したがって上表の0.952を、そのまま実際の処理量が5%減ったと解釈できません。
上表はNNカウンタ追加前の測定履歴です。速度の結論には、以下の追試の合計時間に基づく指標を使います。
推論の処理量と探索完了までの時間を分け、完了したNN評価件数/秒と平均バッチサイズも併記します。

診断用の `nnstats` コマンドは探索を停止・合流した後、完了した推論件数とバッチ呼び出し回数を
`info string nnstats evaluations N batches B` として返します。NNキャッシュのヒットと
終端ノードの再訪問を除き、プリフェッチは含めます。カウンタはバックエンドの生存期間中で累積し、
モデル等の再設定でバックエンドを作り直すとリセットされます。
ベンチマークはウォームアップ後と各探索後の差分を使い、統計取得の通信時間を計時から除きます。

```powershell
python test\build_legacy_probe.py --nnstats --nyugyoku
python test\build_legacy_probe.py --nnstats
python test\evaluate_classic.py benchmark --old OLD_STATS.exe --new NEW.exe --model MODEL.onnx --threads 2 --nodes 20000 --positions 30 --repeats 3 --pv-threads 2 --nn-stats
```

NN評価/秒は評価件数合計/実測時間合計、平均バッチは評価件数合計/バッチ呼び出し回数合計で
計算します。ノード数/NN評価件数と反復別のNN評価速度比も保存します。
これらは推論の処理量の指標であり、探索の有効性や棋力を直接示すものではありません。

NN計測を追加した追試の結果を `test/classic_inference_performance.json` に保存しました。
`python test/collect_classic_performance.py --inference` で再集計できます。
旧版は固定コミット `0cc16854474ab3c7c4f7be66c624f9f9001f7543` から、
EvalNodeの完了推論カウンタと診断コマンドだけを加えて再ビルドしました。
前のNPS表とは異なる実行形式の記録です。前後の表を個々の修正の改善率として比較しません。
この追試でも探索2スレッド・20,000ノード・batch 128・固定30局面×3回を使い、
全360組が指定ノード数に到達しました。

速度比較の主指標は **合計時間に基づくNPS比** とします。
新旧それぞれの合計実ノード数/合計実測時間を求め、その比（新/旧）を比較します。
1より大きければ新版が速いことを表します。実際の完了時間も併記し、時間比（新/旧）は
1より大きければ新版が遅いことを表します。局面ごとのNPS比の中央値は、分布確認用の補助指標です。
ノードの数え方自体が異なるため、主指標も探索の有効性や棋力の優劣を直接示すものではありません。

|モデル|PV dfpnスレッド|探索時間合計・秒（旧→新）|時間比（新/旧）|主指標:合計時間でのNPS比|NN件数/ノードの比（新/旧）|
|:--|--:|--:|--:|--:|--:|
|NY|0|167.979 → 150.979|0.899|1.112|1.053|
|NY|2|160.290 → 151.816|0.947|1.055|1.061|
|no-NY|0|104.473 → 103.096|0.987|1.013|1.077|
|no-NY|2|94.844 → 104.931|1.106|0.904|1.106|

時間は各局面20,000ノード×30局面×3回、計90探索の合計です。
予算超過分を含む実ノード数もJSONに保存しているため、NPS比は時間比の厳密な逆数ではありません。

|モデル|PV dfpnスレッド|NN評価/秒（旧→新）|NN評価速度比|平均バッチ（旧→新）|ノード数/NN件数（旧→新）|
|:--|--:|--:|--:|--:|--:|
|NY|0|6,663 → 7,807|1.172|79.6 → 102.2|1.617 → 1.535|
|NY|2|6,904 → 7,729|1.120|79.7 → 101.9|1.634 → 1.541|
|no-NY|0|11,185 → 12,201|1.091|83.2 → 103.1|1.548 → 1.437|
|no-NY|2|11,885 → 11,888|1.000|82.9 → 102.4|1.603 → 1.450|

NN評価速度は全条件で旧版と同等以上（1.00〜1.17）、平均バッチは約80〜83から約102〜103に増加しました。
一方、新版は実ノードあたりのNN評価件数が約5.3〜10.6%多く、同じノード予算の完了時間比は
0.899〜1.106でした。特に **no-NY・PV 2スレッドでは、94.844秒から104.931秒へ約10.6%長くなっています**。
この時間差を、NN評価速度が同等であることを理由に取り除いて解釈しません。

no-NY・PV 2スレッドのNN評価速度比は1.0002でした。反復別でも1.015、0.991、0.995です。
同じ条件のノード数/NN件数の比は0.9043なので、合計時間によるNPS比は
`1.0002 × 0.9043 ≈ 0.9045` になります。NN処理量がほぼ同じでもNPS比は低く出ます。
この条件の「局面ごとのNPS比の中央値」0.980は補助指標として保存し、速度の結論には主指標の0.9045を使います。
ノード数/NN件数には終端再訪問に加えてキャッシュヒットとプリフェッチ等の差も含まれます。
PV無効でもノードあたりのNN件数には約5.3〜7.7%の差があるため、全差をPV詰み確定ノードの再訪問だけでは説明できません。
キャッシュヒット率やプリフェッチの寄与はまだ分離計測しておらず、原因の内訳は未確定です。
結果の書き込み先がないリクエストを数えればプリフェッチ投入件数は分かりますが、
その結果が後でNNキャッシュ経由で利用される場合もあり、「未使用だった件数」とは区別が必要です。
PV有効時のCPU競合や木のロックの影響も、今回のデータだけでは原因として確定できません。
前の表の0.952をそのまま「NN処理量が5%低下した」とする解釈は撤回します。

pinnedメモリは旧版に合わせて `cudaHostAllocPortable` に変更しました。
今回の実測環境は単一GPUであり、複数GPUでの動作検証は含みません。

### 橋渡し部分の重複処理の削減

`bridge_overhead_fixes.patch` の4点をコードで確認し、取り込みました。
通常のNNリーフ展開では、合法手生成をExtendNodeの1回にまとめ、空判定をルール判定へ渡します。
NN要求は新規辺を方策順に並べ替える前に作られ、プリフェッチ・NNプローブも
`GenerateLegalMoves()` の完全な生成順を渡すため、この順をキャッシュの基準に使えます。
`backend.cpp` にこの契約を明記し、Debugでは全合法手と順序の一致をassertで検査します。
並べ替え後の辺や合法手の部分集合を、そのままNNアダプタへ渡すことはできません。

キャッシュヒットは出力先へ直接書き込み、一時方策配列・並べ替え用配列を廃止しました。
パッチから追加で、特徴量を含むRequestの作成もキャッシュミス後へ移しています。
特徴量のゼロ初期化はミス時に維持し、キャッシュヒットでは不要な生成・初期化を省きます。
Request配列は最初のミスでバッチ上限+32件をreserveし、拡張時の大きな特徴量配列の再コピーを避けます。
32は現在のプリフェッチ設定に合わせた予約量であり、配列の最大件数を制限するものではありません。

探索終了時の`SendMovesStats()`は`VerboseMoveStats`有効時のみ呼びます。
通常実行で破棄されていたNN統計用の履歴再生を省き、明示的な詳細統計出力は維持します。
lc0本体の変更は既存の`search/classic/search.cc`内に収まり、upstreamとの差分は引き続き6ファイルです。
詰み手数の判定、PV dfpnのキー、特徴量からpinnedバッファへのコピーは今回変更していません。

検証結果は`test/bridge_review.json`、再集計は`python test/collect_bridge_review.py`です。

- TensorRT・ONNX Runtime・make_bookの各NY/no-NY、6構成×100局面でNN値・方策・1訪問の上位3手を比較し合格。
- BOOK_POLICYを有効にした定跡生成・読み戻し・価値/方策混合・ウォームアップ後のキャッシュ読み戻しを両モデルで検証。
- 変更前v3との1スレッド・4,000ノード・各30局面の本試験で、最善手・PV・NN件数・NNバッチ数は両モデルとも一致。
- 4構成の2スレッドUSI試験各21項目、Release/Debug単体試験、Debugの合法手assert・キャッシュ試験、multiponderを検証。

実GPUでは非同期の停止判定により、予算超過数や停止時のプリフェッチ件数が揺れる場合があります。
初回の厳密比較で1局面にNN7件の差がありましたが、最善手・PVは一致しました。
変更前バイナリ同士の30局面対照でも訪問数6局面・NN件数1局面に差があり、最善手・PVはすべて一致しました。
各試行の数値を保存し、完全一致を常に保証するとは解釈しません。
`test/check_bridge_equivalence.py --require-exact-nn-counts` でNN件数も厳密な合否条件にできます。

レビュー報告のcallgrind命令数「91k→70k、増分の約64%削減」は、このWindows環境では再計測していません。
今回の証跡は正しさの回帰確認であり、新たなNPS・命令数・棋力の改善率は主張しません。

### 合法手配列の容量予約

`legal_moves_reserve.patch`を取り込み、`ChessBoard::GenerateLegalMoves()`で
cppshogiの`MoveList<Legal>::size()`件を先にreserveするようにしました。
sizeは生成済みの全合法手数であり、指し手の内容・順序・生成回数は変わりません。
変更は`lc0_shogi/position_adapter.cpp`のみで、lc0本体への変更は増えていません。

検証結果は`test/reserve_review.json`、詳細ログと変更前後の比較は`artifacts/reserve_review/`です。
変更前v4のバイナリは`artifacts/bridge_review/`に保存しています。

- 10構成のビルド、Release/Debug単体試験（合法手0件・562件を含む）、Debugのキャッシュ試験に合格。
- TensorRT・ONNX Runtime・make_bookの各NY/no-NY、6構成×100局面のNN比較に合格。
- BOOK_POLICYを有効にした定跡生成・読み戻し・価値/方策混合を両モデルで検証。
- 変更前v4との1スレッド・4,000ノード・各30局面で、最善手・PV・NN評価件数・バッチ数がすべて一致。
  訪問数のみNYの1局面で異なり、停止時の実測値を詳細レポートに残しています。

レビュー報告のcallgrind命令数「25.03億→24.59億、旧版との差を約15%削減」は、
このWindows環境では再計測していません。今回の検証から新たな速度改善率は主張しません。
方策の一時配列、終端リーフでの合法手生成、PV dfpnの文字列キーは変更していません。

### マルチGPUとclassicの並列処理

lc0 `58e78009`と比較し、同じ探索木を共有する点、同時収集数の既定1、
スピン待ち、`MaxPrefetch=32`は共通です。主な差は次のとおりです。

| 項目 | lc0 classic | このUSIエンジン |
|---|---|---|
| 補助スレッド | `TaskWorkers=-1`で自動、workerあたり最大4 | 同じ計算式を利用 |
| 探索worker数 | `Threads=0`ならバックエンド推奨数+1（GPUバックエンド） | `Threads=0`なら有効GPUのスロット総数+1、正数なら指定数 |
| 収集バッチ | roundrobinは各バックエンドの最小値を推奨 | `MinibatchSize=0`なら有効GPUの最大バッチ数の最小値 |
| GPU割り当て | roundrobinは計算作成時にバックエンドを循環指定 | 推論時に空きスロットをFIFOで借用。厳密なroundrobinとは異なる |
| demux/mux | バッチ分割・複数workerのバッチ結合に対応 | 未実装 |

lc0 roundrobinの推奨worker数は構成したバックエンド数なので、GPUごとに1バックエンドなら
自動設定はGPU数+1です。本エンジンで各GPUに1スロットを指定すれば同数になります。
`DNN_Slots*`はGPUの有効化と推論スロット数だけを決め、workerとGPUの固定対応はありません。
`Threads`を正数で指定した場合、スロット数を変えても探索worker数は変わりません。

`ProcessPickedTask()`は補助スレッドと主workerから同じ`AddInput()`を並行呼び出しします。
以前の`std::vector<Request>`への追加は安全でなかったため、vendorの`AtomicVector`を使う
容量検査付きの要求バッファに変更しました。検査はCASで枠を確保してから追加するため、
Releaseでも容量超過は書き込み前に`std::length_error`となります。
容量はGPUの最大バッチ数と明示した収集バッチ数の大きい方+32。
補助タスクの完了待ち後に推論・結果取得し、解放・再利用します。
ONNXの大きな特徴量領域もバックエンド内のプールに残し、バッチごとの再確保を避けます。
キャッシュだけで完結する計算はバッファを借りません。
合法手生成・短手数詰みを含む`EvaluateRules()`は各TaskWorkspaceの独立した履歴を使い、
木の排他ロック外で実行します。終端結果をノードへ反映する短い区間だけ排他にするため、
補助スレッドはこれらのCPU処理も分担できます。

空き推論スロットの取り出しは、末尾から取り出して末尾へ返す方式からFIFOへ変更しました。
逐次推論時に最後のGPUの同じスロットばかり選び続ける偏りを避けます。
TensorRTは各GPUの独立context/stream、ONNXはGPUごとのsessionと推論mutexを使用し、
推論用バッファはスロット単位で排他借用します。
cppshogiのDebug検査`Position::isOK()`の診断用`prevKey`もthread_localにしました。
今回lc0本体への変更は追加していません。

`TaskWorkers=-1`はlc0自動、`MaxConcurrentSearchers=0`は同時収集数無制限です。
並列・バッチ設定は`isready`時に反映します。既定値と単一GPUの測定は末尾を参照してください。

実機用の8通りの比較は次で実行できます。`--old`には`nnstats`計測を組み込んだ旧版を渡します。

```powershell
artifacts/python/python.exe test/check_search_concurrency.py --engine usi/x64/Release/dlshogi_tensorrt.exe --old artifacts/baseline/stats_ny/x64/Release/dlshogi_tensorrt.exe --model artifacts/models/ny.onnx --gpus 0,1 --positions 30 --nodes 20000 --repeats 3 --report artifacts/multigpu_matrix.json
```

1/2スロット/GPU、補助スレッド0/自動、同時収集数1/無制限を比較し、
NN評価/秒・平均バッチ・ノード/秒と各局面の実測値を保存します。
NN統計取得はworkerを停止・合流させてから行い、速度の分母は統計取得と合流を含む経過時間です。
bestmove応答までの時間も別に記録します。従来のNPS表と計測区間が異なる点に注意してください。
`--pv-threads 2`でPV dfpn有効時も確認できます。
ノードの数え方は旧版と異なるため、これだけで探索効率や棋力を判定しません。
同設定で旧版に届かなくても、バックエンド・キャッシュ・CPU処理の差を分離する前に
classic本体の限界とは断定できません。

以下は探索/推論設定の分離前に行った検証です。環境はRTX 4090の1台で、`test/multigpu_review.json`に結果を保存し、
`python test/collect_multigpu_review.py`で再集計できます。

- TensorRT・ONNX・Debugの実バックエンドに8スレッドから96要求を8回投入し、
  直列投入時と値・全合法手の方策を比較。Releaseを含め容量160件の超過例外も確認。
- Release/Debug単体試験で競合する容量超過、要求の欠落・重複、バッファのクリアと再利用を検証。
- TensorRT/ONNXのNY/no-NYで各8設定×6局面、計192探索のbestmove・PV合法性とNN統計を確認。
- 補助スレッド自動・同時収集無制限・探索worker 2で、4構成のUSI試験各21項目に合格。
- make_bookを含む6構成×100局面のNN比較、BOOK_POLICYの生成・読み戻し・混合に合格。
- 変更前の容量予約版との各30局面では最善手・PVが一致。
  NYの2局面で訪問数、うち1局面でNN評価件数に7件の差があり、実測値を保存。
  バッチ数は一致し、no-NYは訪問数・NN評価件数とも一致。
- 10構成をビルドし、通常出力先の実行ファイルとシンボルも更新。

複数GPUの稼働率・負荷分散・速度向上は未検証です。
小規模な単一GPU試験の速度値からマルチGPU構成の推奨値や棋力は判断しません。

### 探索・推論オプションの分離

`UCT_Threads`～`UCT_Threads8`は廃止しました。探索workerとGPUの固定対応はなく、
旧設定の数値を自動変換すると意味が変わるため、互換エイリアスも設けていません。
GUI・スクリプトの設定を次の名前へ更新してください。

| 設定 | 意味 | 既定値 |
|---|---|---|
| `Threads` | 全GPUで共有する探索worker数（0～128）。0はlc0の自動設定（推奨数+1） | 0 |
| `TaskWorkers` | 探索workerごとの補助スレッド数。-1はlc0の自動計算 | -1 |
| `MaxConcurrentSearchers` | 同時にバッチ収集するworker数。0は無制限 | 1 |
| `MinibatchSize` | classicが収集するバッチ数。0は有効GPUの`DNN_Batch_Size*`の最小値 | 0 |
| `DNN_Slots`～`DNN_Slots8` | GPUごとの推論スロット数。0ならそのGPUを使わない | GPU 0のみ2、残り0 |
| `DNN_Model*` | GPUごとのモデル。空欄なら先頭モデルを継承 | 従来どおり |
| `DNN_Batch_Size*` | GPUごとの推論最大バッチ数 | 従来どおり |

バックエンドの推奨worker数は有効GPUのスロット総数です。
したがって既定の単一GPU構成は2スロット・3探索workerになります。
実際のスレッド総数の目安は
「有効な探索worker数 ×（1 + workerごとの補助スレッド数）+ ルートdfpn・PV dfpn・GC・watchdog」です。
USI応答や推論ライブラリのスレッドも別にあります。
`Threads`はプロセス全体のスレッド数ではなく、0の場合は上記の自動設定後のworker数で計算します。
`Threads`を正数で指定した場合はスロット数と独立し、GPUの有効化には使われません。
モデルの指定だけではGPUを有効にしません。無効GPUのモデルファイルは検査・ロードしません。
有効GPUのモデルを開けない場合は、終了前にstdoutへ`info string engine error: ...`を出力します。
明示した`MinibatchSize`がGPUの最大バッチより大きい場合は、要求領域を確保し、
選ばれたGPUの最大バッチ単位で分割推論します。demuxによる複数GPU分割ではありません。
`MinibatchSize`だけを変更した`isready`では要求バッファのプールだけを更新し、
モデル・推論スロット・NNキャッシュを維持します。定跡の方策やキャッシュ容量が変われば、
NNキャッシュと探索木を無効化します。

旧`UCT_Threads*=2`に直接対応する新設定はありません。従来の新版で使った
「全体2 worker・GPU 0に2スロット・補助なし」を再現する設定は次です。

```text
setoption name Threads value 2
setoption name DNN_Slots value 2
setoption name TaskWorkers value 0
isready
```

複数GPUでは、例えば`DNN_Slots=1`・`DNN_Slots2=1`・`Threads=0`で、
2 GPU・計2スロット・3 workerとなります。この構成の速度は実機で確認してください。
単一GPUでは1スロットの自動構成が遅かったため、複数GPUでも各GPUの1スロットと2スロットを比較してください。
旧USI設定は未知オプションとして扱われ、変換されません。
モデル`.ini`の旧設定も`info string Ignoring unknown model option`を出して読み飛ばします。

make_bookとmultiponderも同じ設定体系です。補助エンジンへのオプション文字列も更新が必要です。
selfplay本体は旧探索を使い続け、既存のGPU・スレッド・バッチ引数を維持します。
selfplayのスレッド引数とUSIの`Threads`は意味が異なります。
`--usi_options`で新版USIエンジンを呼ぶ場合は、その文字列だけ新しい名前へ更新してください。
`dlshogi/utils/benchmark.py`も`--threads`（全体）と`--slots`（GPUごと）を分離しました。
旧バイナリを測る場合だけ`--legacy-uct --threads 2`を指定します。

#### 単一GPUでの既定値の選定

RTX 4090・TensorRT・バッチ128・NNキャッシュ100,000件で、まず6構成を
20局面×8,000ノードで予備測定しました。1スロットの自動構成は両モデルで遅く、
2スロットの自動構成を候補に選びました。
続いて固定30局面×20,000ノード×2回を、旧版・従来の新版構成・候補で順番を入れ替えながら測定しました。
Root/PV dfpnは無効で、短手数詰みは有効です。全60試行が各構成で探索予算に到達しました。
時間はNN統計取得とworker合流までを含み、bestmove応答時間も別に保存しています。

| モデル | 構成 | 合計時間（秒） | NN評価/秒 | 平均バッチ | ノード/秒 |
|---|---|---:|---:|---:|---:|
| NY | 旧dlshogi、GPU専用2スレッド | 111.43 | 6,706 | 79.5 | 10,827 |
| NY | 新版、固定2 worker・2スロット・補助0 | 98.37 | 7,989 | 103.1 | 12,258 |
| NY | 新既定、自動3 worker・2スロット・補助自動 | 100.95 | 8,001 | 90.7 | 11,941 |
| no-NY | 旧dlshogi、GPU専用2スレッド | 67.37 | 11,562 | 83.2 | 17,888 |
| no-NY | 新版、固定2 worker・2スロット・補助0 | 63.80 | 13,145 | 105.1 | 18,898 |
| no-NY | 新既定、自動3 worker・2スロット・補助自動 | 52.23 | 16,539 | 99.0 | 23,113 |

新既定の完了時間は固定2 worker構成に対しNYで2.6%長く、no-NYで18.1%短くなりました。
旧dlshogiに対しては、それぞれ9.4%・22.5%短くなりました。
NYでの小幅な低下を含めて記録し、探索側の名前と自動計算をlc0に揃える方針を採っています。
これは当該GPU・TensorRT・局面集合での結果で、棋力やONNX・複数GPUの速度を示すものではありません。

測定用バイナリは`artifacts/options_review/measured_*`に保存しています。
候補値は明示指定して測り、最終バイナリでは既定値の宣言と動作を別途検査します。
`test/tune_search_defaults.py`と`artifacts/options_review/screen_*.json`・`confirm_*.json`に生データを残しています。

移行後の検証結果は`test/search_options_review.json`、再集計は
`python test/collect_search_options.py`です。
10構成のビルド、Release/Debug単体試験、6構成×100局面のNN比較、
4構成×8設定×6局面の192探索、USI試験84項目、移行試験35項目に合格しました。
make_bookの定跡方策混合、multiponderのhit/miss各10回と補助プロセス終了も確認しています。
移行前との各30局面では、補助スレッドなし・1 worker・1スロットに固定し、最善手・PV・NN件数が一致しました。
TensorRT/ONNX/Debugの実バックエンドでは8スレッド同時投入と容量超過を検証し、
収集384件を最大128件の推論3回に分割できることも確認しました。
最大バッチの異なる複数GPUでの実機動作と負荷分散は未検証です。

`isready`の再設定処理を修正した版の検証結果は`test/readiness_review.json`です。
`python test/collect_readiness_review.py`で集計します。10構成のビルド、Release/Debug単体試験、
6構成×100局面のNN比較、7構成の設定・キャッシュ保持試験56項目、USI試験42項目に合格しました。
定跡変更時のキャッシュ無効化、TensorRT/ONNXでの要求プール拡大・縮小・容量超過も確認しています。
性能測定は再実施していません。通常出力先の実行ファイルとシンボルは更新済みです。

### lc0名・単位での探索パラメータ

PUCT・FPU・方策温度の対象12設定はlc0の名前・意味に対応させ、数値はUSI向けの整数`spin`で公開します。
復元した値はlc0の`OptionsParser`で検証し、選択肢・説明もその定義を使います。
USIにはこの一覧だけを公開し、lc0のNN形式・バックエンド・
キャッシュやUCIループは使用しません。宣言と検証は`classic_options.cpp`が橋渡しし、
チェス用の既定値だけを直前版の換算値で上書きします。

| USI設定 | 倍率 | 新版のUSI既定値 | lc0 58e78009の内部既定値（参考） |
|---|---|---|---|
| `CPuct` | 1000 | 2880 | 1.745 |
| `CPuctBase` | 1 | 28288 | 38739 |
| `CPuctFactor` | 1000 | 2000 | 3.894 |
| `CPuctAtRoot` | 1000 | 2320 | 1.745 |
| `CPuctBaseAtRoot` | 1 | 25617 | 38739 |
| `CPuctFactorAtRoot` | 1000 | 2000 | 3.894 |
| `RootHasOwnCpuctParams` | — | true | false |
| `FpuStrategy` | — | reduction | reduction |
| `FpuValue` | 1000 | 540 | 0.330 |
| `FpuStrategyAtRoot` | — | reduction | same |
| `FpuValueAtRoot` | 1000 | 0 | 1（sameの間は無視） |
| `PolicyTemperature` | 1000 | 1740 | 1.359 |

`RootHasOwnCpuctParams=false`ではルートも非ルートのPUCT設定を使います。
`FpuStrategyAtRoot=same`では戦略と値を両方継承し、`FpuValueAtRoot`を無視します。
数値はすべて`spin`、戦略は`combo`、ルートの独立設定は`check`です。
たとえば`CPuct=2880`は内部の2.88、`FpuValue=-1000`は内部の-1に対応します。
小数・指数表記・不正文字列・範囲外の入力は拒否し、直前の値を保持します。
数値の整数範囲は橋渡し層に定義し、`test/check_scaled_option_ranges.py`で
固定コミットのlc0範囲との一致を検査します。内部へ復元した値もupstreamで検証します。
`setoption`は次の`isready`で探索用に確定するため、進行中の探索設定には混入しません。

これら12設定は`.ini`専用ではなく、すべてUSIオプションとして設定できます。変更例:

```text
setoption name CPuct value 3000
setoption name CPuctFactor value 2000
setoption name FpuStrategy value reduction
setoption name FpuValue value 600
setoption name PolicyTemperature value 1800
isready
```

小数指定などを拒否した場合は`info string Rejected option ...`で理由を表示し、
1000分率の項目には`specify an integer in 1/1000 units`も表示します。直前の値は保持します。

`PolicyTemperature`はNNのlogitを割る方策温度です。dlshogiバックエンドで適用し、
classic側に返した確率へ再適用しません。lc0の`Temperature`は着手選択の温度で、
これとは別物です。`Random_*`・`Random2_*`をlc0の温度設定へ置き換えてはいません。
`Draw_Value_*`・`Draw_Ply`・詰み探索・`Eval_Coef`・`DNN_*`・定跡設定もdlshogi固有のままです。

旧名はUSIの互換エイリアスを設けず廃止しました。モデル`.ini`内にあれば
`info string Ignoring unknown model option`で警告して読み飛ばします。
旧版でも同じ`.ini`を使う場合は、旧キーを残して次の新キーを追加してください。

| 旧キー | 新キーへの係数換算 |
|---|---|
| `C_init` | `CPuct = 値 × 20` |
| `C_base` | `CPuctBase = 値` |
| `C_init_root` | `CPuctAtRoot = 値 × 20` |
| `C_base_root` | `CPuctBaseAtRoot = 値` |
| 固定していた探索項の対数係数 | `CPuctFactor=2000`、`CPuctFactorAtRoot=2000` |
| ルートの独立設定 | `RootHasOwnCpuctParams=true` |
| `C_fpu_reduction*` | `FpuValue* = 値 × 20`、戦略は`reduction`（挙動の互換性はない） |
| `Softmax_Temperature` | `PolicyTemperature = 値 × 10` |

これは係数の単位換算で、旧版と同じ探索を保証するものではありません。
旧FPUは同じ手を選び直しても方策を累積し、勝率0で下限を切ります。
classicは異なる訪問済みの手の方策和を使い、同じ下限処理もありません。
`FpuValue`は対局で再調整する前提で、USI既定540（内部0.54）は現状維持の暫定値です。
`reduction / 540`、より大きい減算量、`absolute / -1000`などを、ルート設定を固定し、
同じ開始局面・持ち時間・モデルで先後を入れ替えて比較してください。今回、棋力調整は行っていません。

追記版の作成には次のスクリプトを使えます。元ファイルと既存の新キーは上書きしません。
既存値が新しいlc0の範囲に入らない場合は、エンジンの読み込み結果を確認して調整してください。

```text
python dlshogi/utils/migrate_classic_ini.py model.onnx.ini --output model.onnx.ini.new
```

前回の小数形式の新キーが既にある場合は、明示的に`--from-float-options`を追加します。
この指定は既存の新数値キーを整数化します。整数化済みファイルへ再指定しないでください。
1000分率で表せない小数は、勝手に丸めずエラーにします。
`usi_params_optimizer.py`も新名・整数単位へ移行しました。
selfplay本体は旧探索を維持し、CLIの`--c_fpu_reduction`なども変更していません。

FPU名のみを移行した前段階の結果は`test/fpu_review.json`に保存しています。

前回の小数形式での全面移行の記録は`test/native_options_review.json`です。
この版と次の整数化版では`ReuseSubtree=false`の反映漏れがあり、当時の30局面比較は
前提を満たしていません。以下は当時の実測記録で、比較の検証根拠は後述の修正版へ差し替えます。
その時点では10構成のビルド、Release/Debug単体試験、
NN比較600局面、移行・設定試験91項目、USI試験42項目に合格しました。
前回版との両モデル各30局面で最善手・PV・NN件数・バッチ数が一致し、NYの1局面だけ訪問数に差がありました。
定跡方策とmultiponderも確認し、通常出力先の実行ファイル・シンボルを更新しています。
速度・棋力の再測定は行っていません。

指定された`H:\model\model-pre68_40x512-009.onnx.ini`と
`H:\model\model-pre44_resnet30x384_relu_b4096lr004-013.onnx.ini`には、旧キーを残して12キーを追記しました。
今回の整数化後の共通値は`CPuct=2540`、`CPuctBase=27126`、`CPuctFactor=2000`、
ルート`2240 / 33311 / 2000`、独立ルート有効、FPUは`reduction / 620`、ルート`reduction / 0`です。
`PolicyTemperature`はpre68が1600、pre44が1400です。旧dlshogiの6キーの元のバイト列を保持していることを照合し、
今回の変更直前のバックアップを`artifacts/integer_options/ini/*.before`へ保存しています。
旧版・新版のRelease双方で追記版を読み込み、同じ小型の検証用NNで方策・価値の一致を確認しました。
これは設定ファイルの互換性試験で、指定された大型モデル本体の推論・性能試験ではありません。

整数`spin`へ変更した現在の結果は`test/integer_options_review.json`、再集計は
`python test/collect_integer_options_review.py`です。10構成のビルド、Release/Debug単体試験、
NN比較600局面、移行・設定試験91項目、USI試験42項目、整数範囲9項目に合格しました。
前回版との両モデル各30局面で最善手・PV・NN件数・バッチ数が一致しました。
no-NYの3局面では訪問数だけに差があり、記録に保存しています。
通常出力先の実行ファイル・シンボルと、指定された2つの`.ini`を整数版へ更新済みです。

### ReuseSubtreeの反映漏れの修正

通常USIの`isready`で`SetReuseSubtree`を呼ぶ処理を復元しました。
新オプションへの移行時に隣接する旧パラメータ処理と一緒に削除していた不具合です。
`test/check_ready_settings.py`は12設定をすべてUSI経由で変更し、
`ReuseSubtree`・表示・評価値換算・ランダム着手・探索並列設定など計45値を実際のセッションから読み戻します。
指定値の写しではなく、探索へ渡すパラメータと実行時の変数を検査します。
これに加えて、3,000ノード探索後の最善手を指した局面を1ノード探索し、
再利用ありで開始時の訪問数が残り、再利用なしで0になることを確認します。

診断コマンド`searchsettings`は探索を停止・合流してから`info string searchsettings {JSON}`を返します。
CPuct・FPU・方策温度や勝率などは内部単位で表示します。USIの`setoption`では引き続き整数の倍率を使ってください。

修正後の検証結果は`test/reuse_fix_review.json`、再集計は`python test/collect_reuse_fix_review.py`です。
比較基準には再利用無効化が正常な`artifacts/fpu_review`のバイナリを使い、
両モデル各30局面で最善手・PV・NN件数・バッチ数が一致しました。
NYの1局面で訪問数だけに差があり、全記録に保存しています。
8構成で設定読み戻し・木の再利用・拒否理由の試験を通し、ほかに10構成のビルド、
Release/Debug単体試験、NN比較600局面、USI試験42項目、定跡・multiponderの回帰試験も再実行しました。
実行ファイル・シンボルは更新済みです。2つのモデル`.ini`は正しい整数値のまま変更していません。
