# polars_mc

[Polars](https://pola.rs/) と NumPy を基盤にした、**ベクトル化モンテカルロ・シミュレーション**ライブラリ。

利用者が書くのは **1試行ぶんの普通の関数だけ**。乱数の再現性・チャンク分割・並列評価・チャンク横断の集計合成は実行基盤が引き受けます。さらにその基盤自体が、John Hughes「[なぜ関数プログラミングは重要か](https://www.sampou.org/haskell/article/whyfp.html)」の遅延ストリーム合成として組み立てられており、**どの層も単独で使えます**。

```python
from polars_mc import Simulation, Normal

def trial(width, height):                      # 1試行を普通の関数で書く
    area = width * height
    return {"area": area, "passed": area >= 48.0}

sim = Simulation(
    inputs={"width": Normal(10.0, 0.2), "height": Normal(5.0, 0.1)},
    trial=trial,
    outputs={"area": ["mean", "std", ("q", 0.95)], "passed": ["mean"]},
)
print(sim.run(1_000_000, seed=42).summary())   # 100万試行、メモリ一定
```

## 3つの層 — 好きな高さから使う

このライブラリは「1つの入口」ではなく、**独立して使える3層**として設計されています。上の層は下の層の薄い合成にすぎず、下の層だけを取り出して使っても構いません。

| 層 | モジュール | 単独で使うと | 依存 |
|---|---|---|---|
| **① 合成コア** | `stream.py` | 遅延ストリームの汎用コンビネータ。ニュートン法・数値微分・数値積分など、モンテカルロと無関係な収束計算にも使える | 純 Python |
| **② サンプリング／推定器** | `sampling.py` / `estimate.py` | 分布 → 再現可能な無限サンプル列。`expectation` / `probability` / `integrate` / `estimate_pi` で「目標精度に達したら止まる」推定 | + NumPy |
| **③ シミュレーション基盤** | `engine.py` ほか | 多入力・多出力の `Simulation`。固定回数の `run` と適応停止の `run_until` | + Polars |

```python
from polars_mc import Stream            # ① だけ使う
from polars_mc import sample_stream     # ② だけ使う
from polars_mc import Simulation        # ③ フルスタック
```

③ は ② の上に、② は ① の上に載っているだけで、逆向きの依存はありません。② と ③ は乱数のシード生成も共有しています — `rng.generator_stream(seed)` という「チャンクごとの独立乱数生成器の無限 `Stream`」が唯一の種であり、② はそれを `map` してサンプル配列にし、③ はそれを `map` して実行済みチャンクにします。ストリームを流れるのは**乱数の種であってサンプルデータではない**ので、`backend="processes"` でも小さな生成器だけがワーカーに渡り、巨大な行列はプロセス境界を越えません。

## 特長

- **書くのは1試行だけ** — 普通の Python（NumPy）関数として書けば、ブロードキャストで全試行ぶんが一気にベクトル化実行される。
- **完全な再現性** — 乱数はチャンクごとに独立シード。同一シードなら、チャンクサイズ・バックエンド・ワーカー数に関係なくビット単位で同一結果。
- **メモリ有界** — 何億試行でも、各チャンクは小さな集計だけを返すので一定メモリで回る。
- **数値的に安定な合成** — 平均・分散は Chan の並列アルゴリズムでチャンク横断合成。
- **2つの並列バックエンド** — 軽い計算は Polars 内蔵スレッド、重い CPU 処理（Shapely 等）はプロセス並列。
- **固定回数でも適応停止でも** — `run(n_trials)` と `run_until(target_se=...)` は同じ畳み込みコアを共有し、終端条件だけが違う。

## インストール

[uv](https://docs.astral.sh/uv/) でこのリポジトリを使う場合:

```bash
uv sync                 # 実行用 + 開発用の依存
```

別プロジェクトから依存に加える場合（ローカルパス例）:

```bash
uv add path/to/polars-mc
```

ランタイム依存は `polars` と `numpy` だけです。Python 3.12 以上。

---

# ① 合成コア: `Stream`

`Stream` は **再実行可能な遅延列**（一度きりのイテレータではなく「値」）です。NumPy にも Polars にも依存しない純 Python で、記事が言う2種類の「のり」をそのまま提供します。

- **高階関数**: `map` / `filter` / `scan`（running fold）/ `reduce`（foldl）/ `take` / `drop` / `take_while` / `take_until` / `pairwise` / `zip_with`
- **無限列と収束判定**: `Stream.iterate(f, x0)`（記事の `repeat`）、終端の `within(eps)` / `relative(eps)` / `first()` / `to_list()`

記事のニュートン法による平方根は、そのまま1行になります。

```python
from polars_mc import Stream

Stream.iterate(lambda x: (x + 2 / x) / 2, 1.0).within(1e-12)   # 1.414213562373095
```

「無限の近似列を作り、十分近づいたら打ち切る」という骨格は、モンテカルロにも数値積分にも共通です。数値微分・台形則積分まで含めた記事の再現は [`examples/whyfp_numeric.py`](examples/whyfp_numeric.py) にあります。

---

# ② サンプリングと推定器

## 分布 → 無限サンプル列

`sample_stream` は分布を、`input_stream` は複数入力の辞書を、**再現可能な無限チャンク列**に変えます。k 番目のチャンクは、最終的に何チャンク消費したかに関係なく常に同じ配列です。

```python
import numpy as np
from polars_mc import sample_stream, Normal

# 標準正規で X > 2 となる割合を、チャンクごとに眺める
rates = (
    sample_stream(Normal(0, 1), seed=7, chunk=10_000)
    .map(lambda a: float((a > 2.0).mean()))
    .take(5)
    .to_list()
)
# [0.0219, 0.0229, 0.022, 0.0232, 0.0209]
```

## 推定器コンビネータ

モンテカルロ推定はどれも「乱数の無限列 → 推定量に `map` → running 平均に `scan` → 標準誤差で停止」という同じ骨格です。よく使う形を部品として用意しています（いずれも `Estimate` を返す）。

```python
import numpy as np
from polars_mc import estimate_pi, integrate, expectation, probability, Normal

estimate_pi(target_se=1e-3)
# Estimate(value=3.14189, standard_error=0.000999, n_samples=2,700,000, converged)

integrate(lambda x: np.sin(x), 0.0, np.pi, target_se=1e-3)      # ∫₀^π sin x dx ≈ 2
# Estimate(value=1.9999, standard_error=0.000997, n_samples=940,000, converged)

expectation(lambda x: x**2, Normal(0, 1), target_se=2e-3)       # E[X²] ≈ 1
probability(lambda x: x > 0, Normal(0, 1), target_se=1e-3)      # P(X>0) ≈ 0.5
```

停止条件は3種類あり、いずれも `Estimate.converged` に結果が出ます。

| 引数 | 停止条件 |
|---|---|
| `target_se=eps` | 平均の標準誤差が `eps` 以下になったら停止 |
| `within=eps` | 連続する推定値の差が `eps` 以下（記事の `within`） |
| `relative=eps` | 同じく相対誤差版（記事の `relative`） |
| `max_trials=N` | 到達しなければ N 試行で打ち切り、`converged=False` |

自前の推定量を同じ骨格で回したいときは、チャンク列を作って `estimate_mean` に渡すだけです。

```python
from polars_mc import estimate_mean, sample_stream, Exponential

chunks = sample_stream(Exponential(2.0), seed=0).map(lambda a: np.minimum(a, 5.0))
est = estimate_mean(chunks, target_se=1e-3)     # E[min(X, 5)] を SE 1e-3 まで
```

`Estimate` は `value` / `standard_error` / `n_samples` / `converged` を持つ frozen dataclass です。

---

# ③ シミュレーション基盤: `Simulation`

多入力・多出力で、列ごとに統計量を宣言して回す層です。

## クイックスタート

製造される長方形の面積をモンテカルロし、規格（面積 ≥ 48）の歩留まりを求める例:

```python
import numpy as np
from polars_mc import Simulation, Normal

# 1) 1試行を普通の関数として書く（width, height は NumPy 配列で渡ってくる）
def trial(width, height):
    area = width * height
    passed = area >= 48.0
    return {"area": area, "passed": passed}

# 2) 入力分布・trial・集計したい統計量を宣言
sim = Simulation(
    inputs={"width": Normal(10.0, 0.2), "height": Normal(5.0, 0.1)},
    trial=trial,
    outputs={
        "area":   ["mean", "std", "min", "max", ("q", 0.95)],
        "passed": ["mean"],   # bool 列の平均 ＝ 合格率
    },
)

# 3) 実行して読む
result = sim.run(n_trials=1_000_000, seed=42)
print(result.summary())
print(result.value("area", "mean"))      # 49.9997
print(result.value("passed", "mean"))    # 0.9224（合格率）
```

`result.summary()` の出力:

```
Monte Carlo simulation: 1,000,000 trials in 4 chunk(s), seed=42
  area (quantiles approximate):
    mean     = 49.9997
    std      = 1.41349
    min      = 43.5252
    max      = 57.0375
    q0.95    = 52.352
  passed:
    mean     = 0.922392
```

## trial の書き方

2つのスタイルがあり、エンジンが関数のシグネチャから自動判定します（`trial_style=` で明示も可）。引数名が入力名と一致すれば **array**、`(df)` のように1つの frame を取れば **frame** です。

### array スタイル — 普通の関数（推奨）

入力は NumPy 配列（チャンクぶんまとめて）として渡され、`{出力列名: 配列}` を返します。NumPy がブロードキャストするので、「1試行ぶんのつもりで書いた普通のコード」がそのままフルにベクトル化されます。

```python
def trial(width, height):
    area = width * height
    return {"area": area, "passed": area >= 48.0}
```

- 引数は **必要な入力名だけ** 書けばよい（使わない入力は省略可）。
- 返り値は `dict[str, 配列]`。長さ `n` の配列、またはスカラ（自動ブロードキャスト）。
- 入力列もそのまま集計対象にできる（`outputs` に入力名を書ける）。
- **スカラ前提の `if` は使えない** → `np.where` / `np.select` / `np.clip` などで書く。

```python
grade = np.where(area >= 50.0, 1.0, 2.0)     # 条件分岐は np.where で
```

### frame スタイル — Polars 式

Polars の遅延最適化を効かせたい、または NumPy では書きにくいパイプライン向け:

```python
import polars as pl

def trial(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        area=pl.col("width") * pl.col("height"),
    ).with_columns(
        passed=pl.col("area") >= 48.0,
    )
```

## 入力分布

`inputs` には列名と分布を渡します。乱数生成は NumPy に委譲されます。

| 分布 | 引数 | 意味 |
|---|---|---|
| `Normal(mean, std)` | 平均・標準偏差 | 正規分布 |
| `Uniform(low, high)` | 下限・上限 | 連続一様分布 `[low, high)` |
| `LogNormal(mu, sigma)` | 基準正規の平均・標準偏差 | 対数正規分布（`log(X) ~ Normal(mu, sigma)`） |
| `Triangular(left, mode, right)` | 最小・最頻・最大 | 三角分布（`left ≤ mode ≤ right`） |
| `Exponential(scale)` | スケール（= 1/レート） | 指数分布 |
| `Bernoulli(p)` | 成功確率 | ベルヌーイ分布（`True`/`False` を返す） |
| `Constant(value)` | 定数 | 全試行同じ値（固定パラメータを列として持たせる用） |

### 独自分布を追加する

`Distribution` を継承し、`sample(rng, n)` で長さ `n` の NumPy 配列を返すだけです。`Simulation` でも `sample_stream` でも、同じ分布クラスがそのまま使えます。

```python
import numpy as np
from polars_mc import Distribution

class Categorical(Distribution):
    """重み付きでカテゴリ（整数コード）を選ぶ離散分布。"""
    def __init__(self, weights):
        w = np.asarray(weights, dtype=float)
        self.p = w / w.sum()

    def sample(self, rng: np.random.Generator, n: int):
        return rng.choice(len(self.p), size=n, p=self.p)
```

`rng` はチャンクごとに独立シード済みの `numpy.random.Generator` です。これだけを使って引けば再現性が保たれます。

## 出力と統計量

`outputs` は `{列名: [統計量, ...]}`。列は trial が生成した列でも、入力列でも構いません。

| 統計量 | 内容 |
|---|---|
| `"count"` | 件数 |
| `"sum"` | 合計 |
| `"mean"` | 平均 |
| `"var"` | 分散（不偏、ddof=1） |
| `"std"` | 標準偏差 |
| `"min"` / `"max"` | 最小 / 最大 |
| `"median"` | 中央値（`("q", 0.5)` と同じ） |
| `("q", p)` | `p` 分位点（例: `("q", 0.95)`） |

bool 列の `"mean"` は **True の割合**（合格率・確率）になります。

### 分位点だけは近似になりうる

平均・分散・min・max・count は**チャンク横断で厳密に合成**できます。一方 **分位点は加法的に合成できない**ため、列ごとに一様サブサンプルを上限つき（`quantile_sample_size`、既定 200,000）で保持し、その標本から推定します。

- 総試行数が `quantile_sample_size` 以内 → 全データ保持で **厳密**。
- 予算を超える → **近似**。その列名が `result.approximate_columns` に入り、`summary()` にも `(quantiles approximate)` と表示される。

厳密にしたければ `quantile_sample_size` を総試行数以上にしてください（その分メモリを使います）。

## 結果を読む（`SimulationResult`）

```python
result = sim.run(1_000_000, seed=42)

result.value("area", "mean")      # 単一の統計量 -> float
result["area"]                    # その列の全統計量 -> {"mean": ..., "std": ...}
result.to_dict()                  # {列: {統計量: 値}}
result.to_polars()                # column / statistic / value の縦持ち DataFrame
print(result.summary())           # 人が読む複数行サマリ（repr も同じ）

result.approximate_columns        # 分位点が近似になった列名の集合
result.n_trials, result.n_chunks, result.seed
result.converged, result.standard_error, result.target   # run_until のときだけ非 None
```

`to_polars()` は後段の分析・保存に便利です（例: `result.to_polars().write_csv("out.csv")`）。

```
shape: (6, 3)
┌────────┬───────────┬───────────┐
│ column ┆ statistic ┆ value     │
╞════════╪═══════════╪═══════════╡
│ area   ┆ mean      ┆ 49.999715 │
│ area   ┆ std       ┆ 1.413486  │
│ area   ┆ q0.95     ┆ 52.351952 │
│ passed ┆ mean      ┆ 0.922392  │
└────────┴───────────┴───────────┘
```

## 実行オプション（`sim.run`）

```python
sim.run(
    n_trials,             # 総試行回数
    *,
    chunk_size=None,      # 1チャンクの行数。None なら min(n_trials, 250_000)
    seed=0,               # マスターシード
    backend="sequential", # "sequential" | "processes"
    n_workers=None,       # processes 時のワーカー数（既定: 全CPU）
)
```

### バックエンドの選び方

| trial の中身 | backend | 理由 |
|---|---|---|
| 算術・補間・VLOOKUP など軽い処理 | `"sequential"`（既定） | Polars 内蔵スレッドで十分速い |
| Shapely 等の重い CPU 処理 | `"processes"` | チャンクが独立した CPU 作業。コア数にほぼ比例して高速化 |

```python
result = sim.run(1_000_000, seed=42, backend="processes", n_workers=8)
```

`backend="processes"` は **ピクル可能な trial** が必要です（モジュールトップレベル関数にする。ラムダ・局所関数は不可）。スクリプトのエントリは `if __name__ == "__main__":` でガードしてください。再現性はバックエンドに依存しません（同一シードなら逐次と完全一致）。

### チャンクサイズの指針

ベクトル化版では、チャンクサイズの基準は処理時間ではなく **メモリとスレッド飽和** です。1チャンク 10万〜100万行を目安に、`行数 × 列数` が RAM／キャッシュに収まる範囲で調整してください。未指定なら `min(n_trials, 250_000)`。

`backend="processes"` では、チャンク数がワーカー数より十分多いほど負荷が均等に分散します。チャンクが大きすぎる（＝チャンク数が少ない）と並列度が出ません。

## 収束するまで回す（`run_until`）

`n_trials` を決め打ちする代わりに、**指定した出力列の平均が目標精度に達したら止める**こともできます。停止条件は推定器と同じ（`target_se` / `within` / `relative` / `max_trials`）。

```python
# area の平均が標準誤差 1e-3 に達するまで回す（他の出力もその時点で集計）
result = sim.run_until("area", target_se=1e-3, seed=42)
print(result.summary())
```

```
Monte Carlo simulation: 2,000,000 trials in 200 chunk(s), seed=42 [adaptive on 'area': converged, se=0.001]
  area (quantiles approximate):
    mean     = 49.9996
    std      = 1.41392
    ...
```

`run`（固定回数）と `run_until`（適応停止）は同じチャンク畳み込みコア（`Moments` モノイドの合成）を共有し、終端が「N 個取る」か「収束まで取る」かだけが違います。各チャンクは i.i.d. なので、途中で止めても分位点は健全に推定できます（予算超過時は近似フラグが立つ）。

---

## サンプル集

| ファイル | 内容 |
|---|---|
| [`examples/rectangle.py`](examples/rectangle.py) | 基本。歩留まり計算（array スタイル） |
| [`examples/lookup.py`](examples/lookup.py) | テーブル参照／補間（VLOOKUP 的処理）。`np.interp` / `np.searchsorted` / ファンシーインデックス |
| [`examples/shapely_intersection.py`](examples/shapely_intersection.py) | Shapely の幾何交差。`backend="processes"` の効果も計測 |
| [`examples/ellipse_intersection.py`](examples/ellipse_intersection.py) | 多角形近似した楕円同士の接触面積。交差多角形の頂点数が閾値以下の試行だけ近似頂点数を2倍にして再計算（適応細分化）＋ `run_until` で自動停止 |
| [`examples/whyfp_numeric.py`](examples/whyfp_numeric.py) | 記事の数値例（平方根・数値微分・数値積分）を `Stream` コンビネータだけで再現 |
| [`examples/pi.py`](examples/pi.py) | π の適応モンテカルロ推定 |
| [`examples/integrate.py`](examples/integrate.py) | モンテカルロ数値積分（目標標準誤差で自動停止） |

```bash
uv run python examples/rectangle.py
uv run python examples/lookup.py
uv run python examples/whyfp_numeric.py
uv run python examples/pi.py
uv run python examples/integrate.py
uv run --with shapely python examples/shapely_intersection.py
uv run --with shapely python examples/ellipse_intersection.py
```

## 設計

```
                                     ← 利用者が書くのはここだけ
モデル層     （利用者の trial 関数）    1行＝1試行

③ 実行層     engine.py          チャンク計画・並列実行・合成（run / run_until）
   集計層     aggregate.py       Moments モノイド合成 + 分位点サブサンプリング
   チャンク層 chunk.py           入力サンプリング → trial 適用 → 部分集計
   結果層     result.py          SimulationResult（summary / to_dict / to_polars）

② 推定器層   estimate.py        expectation / probability / integrate / estimate_pi
   ストリーム層 sampling.py      分布 → 再現可能なサンプル無限列
   入力仕様層 distributions.py   Normal / Uniform / LogNormal / ...

   RNG層      rng.py             seed → チャンク別の独立乱数生成器の Stream（②③ が共有）
① 合成コア   stream.py          遅延コンビネータ（map / scan / take_until / within / …）
```

**なぜこの形か。** `stream.py` は記事「なぜ関数プログラミングは重要か」のエッセンス（高階関数と遅延無限列）を純 Python で表した合成コアです。`engine.py` の `run`（batch）も `run_until`（適応停止）も、このコアの上で `aggregate.Moments` のモノイド合成をチャンク列に畳み込むだけの実装になっています。だから両者の差は「終端条件」だけで済み、`estimate.py` の推定器群も同じ骨格の別の貼り合わせ方として書けます。

**なぜ分位点だけ特別扱いか。** チャンクをまたいで厳密に合成できるのは**加法的な統計量**だけです。エンジンは列ごとに count / mean / M2（偏差平方和）/ min / max を保持し、数値的に安定な Chan の並列アルゴリズムで合成します。分位点は加法的でないのでサブサンプル推定にしています（→「分位点だけは近似になりうる」）。

**なぜチャンクループは逐次か。** チャンクを順に畳み込むことでメモリが有界に保たれ、各チャンクが独立シードを持つことで結果が完全に再現します。並列化は、既定では Polars 内蔵スレッドプール（チャンク内の式評価）に、`backend="processes"` ではチャンク単位のプロセス分散に委譲します。

**なぜシード生成が1箇所か。** `run`（一括spawn）も `run_until`（遅延spawn）も `sample_stream` も、種はすべて `rng.generator_stream` 由来です。`SeedSequence` は生成した子の数を内部に持つため、一括 spawn と1個ずつの遅延 spawn は同一の子シード列を返します。だからこの一本化は既存シードの結果をビット単位で変えず（実測で確認済み）、シードロジックの重複もありません。

## 注意点

- array スタイルでは `if width > 5:` のようなスカラ前提の分岐は配列に対して動きません。`np.where(width > 5, a, b)` と書いてください。算術・比較・NumPy 関数で書く限り、普通の関数のままフル速度で動きます。
- 試行ごとに発散する制御フローや、スカラ専用の外部ソルバを1試行ごとに呼ぶ処理は、ベクトル化の守備範囲外です（Shapely のように配列対応 API があるものは別）。
- Windows のコンソール（cp932）で `to_polars()` の結果を `print` すると、罫線文字でエンコードエラーになることがあります。ライブラリの問題ではなく端末の文字コードの問題なので、`PYTHONIOENCODING=utf-8` を設定してください。

## 開発

```bash
uv sync               # 実行用 + 開発用の依存
uv run pytest         # テスト（80件）
uv run ruff check .   # リント
uv run mypy src       # 型チェック（strict）
```
