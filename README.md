# OpenMP Matrix Multiplication Benchmark

CとOpenMPによる行列積の性能検証プロジェクトです。CPUキャッシュを意識したループ順序の変更と、OpenMPによる並列化が処理時間に与える影響を測定します。

現在は授業課題で使用した初期実装を、ほぼ変更せず保存した段階です。今後、比較条件と正確性検証を整備し、再現可能なベンチマークへ発展させます。

## 現在の実装

`src/main.c` には次の2種類の行列積があります。

- `matmul`: `i-j-k` 順の逐次実装
- `mymatmul`: `i-k-j` 順への変更と、外側ループのOpenMP並列化を適用した実装

## コードの由来

初期版のうち、自分で実装した部分は `mymatmul` です。`matmul`、計測、結果比較、`main` などの部分は授業で提供されたコードを基にしています。今後追加する実装の分離、測定の自動化、正確性検証、グラフ生成および考察は、このリポジトリで行った変更としてGit履歴に残します。

## 必要な環境

- C11をコンパイルできるGCC
- OpenMP
- CMake 3.20以降
- MinGW Makefilesに対応したビルドツール（Windowsの場合）

動作確認に使用した環境は次のとおりです。

- Windows 11
- GCC 14.2.0 (MinGW-w64)
- CMake 4.4.2
- GNU Make 4.4.1

## ビルド

PowerShellでリポジトリのルートから実行します。

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`-O3` は強いコンパイラ最適化を有効にし、`-march=native` は実行するCPU向けの命令を利用できるようにします。OpenMPのコンパイル・リンク設定はCMakeが行います。

WindowsのMinGW Makefilesは、日本語を含むパスで `Illegal byte sequence` になる場合があります。その場合は、リポジトリを `C:\dev\openmp-matmul-benchmark` など英数字だけのパスにcloneしてビルドしてください。

## 実行

引数に正方行列の一辺の要素数を指定します。

```powershell
.\build\matmul_benchmark.exe 1000
```

プログラムは逐次実装を基準として計算結果を比較し、正しければ処理時間、GFLOPS、逐次実装に対する高速化率を表示します。

スレッド数は環境変数 `OMP_NUM_THREADS` で指定できます。

```powershell
$env:OMP_NUM_THREADS = 4
.\build\matmul_benchmark.exe 1000
```

## 現時点の制約

- ループ順序変更とOpenMP並列化を同時に適用しているため、個々の効果を分離できない
- 各実装を1回しか測定しておらず、実行時間のばらつきを評価していない
- `CLOCK_REALTIME` を使用しており、性能測定用の時計として改善の余地がある
- 基準値がゼロの要素に対する誤差検査が不十分
- コマンドライン引数とメモリ確保のエラー処理がない

## 今後の完成条件

1. `i-j-k` 逐次版、`i-k-j` 逐次版、`i-k-j` OpenMP版を分離して比較する
2. 行列サイズとスレッド数を変え、各条件を複数回測定する
3. ゼロを含む入力でも正しさを検証する
4. 実行環境、ビルド方法、測定条件を記録する
5. 結果をグラフ化し、高速化の理由と限界を考察する

ブロッキングやGPU化は本プロジェクトの必須範囲には含めません。

