# polars_mc

[Polars](https://pola.rs/) を基盤にしたベクトル化モンテカルロ・シミュレーション。

利用者が書くのは **1試行（1行＝1試行）だけ**。再現可能な乱数サンプリング・チャンク分割・Polarsによる並列評価・チャンク横断の集計合成は、すべて実行基盤が隠蔽します。モデル作成者はチャンク・スレッド・乱数シードを一切意識する必要がありません。

## 設計

```
入力仕様層   distributions.py   Normal / Uniform / LogNormal / ...
モデル層     （利用者のtrial関数）  1行＝1試行          ← 利用者が書くのはここだけ
RNG層        rng.py             seed → チャンク別の独立な乱数生成器
チャンク層   chunk.py           入力サンプリング → trial適用 → 部分集計
集計層       aggregate.py       加法的モーメント合成 + 分位点サブサンプリング
実行層       engine.py          チャンク計画・Polars並列実行・合成
結果層       result.py          SimulationResult（summary / to_dict / to_polars）
```

チャンクをまたいで厳密に合成できるのは **加法的な統計量** だけです。そのためエンジンは列ごとに count / mean / M2（偏差平方和）/ min / max を保持し、数値的に安定な Chan の並列アルゴリズムで合成します。

**分位点は加法的に合成できません。** そこで列ごとに一様サブサンプルを上限つきで保持し、その標本から推定します。全試行が `quantile_sample_size` の予算内に収まる場合、分位点は厳密値になります。予算を超えた場合は近似となり、その列名が `result.approximate_columns` に記録されます。

並列化は Polars 内蔵のスレッドプールに委譲します（各チャンク内の式評価を並列化）。チャンクループ自体は逐次実行で、これによりメモリが有界に保たれ、同一シードでの結果が完全に再現します。より重いバックエンド（multiprocessing、Dask）は、この API を変えずに後から追加できます。

## 使い方

trial は2つのスタイルのどちらでも書けます。エンジンが関数のシグネチャから自動判定します（`trial_style=` で明示指定も可能）。

### array スタイル — 普通の関数（推奨）

入力は NumPy 配列として（チャンクぶんをまとめて）渡されます。NumPy がブロードキャストするため、「1試行ぶんを書いたつもりの普通のコード」がそのまま全試行ぶん完全にベクトル化されて実行されます。スカラ前提の `if` 分岐の代わりに `np.where` / `np.clip` などを使ってください。

```python
import numpy as np
from polars_mc import Simulation, Normal

def trial(width, height):              # 1行＝1試行。Polars式は不要
    area = width * height
    passed = area >= 48.0
    return {"area": area, "passed": passed}

sim = Simulation(
    inputs={"width": Normal(10.0, 0.2), "height": Normal(5.0, 0.1)},
    trial=trial,
    outputs={
        "area":   ["mean", "std", "min", "max", ("q", 0.95)],
        "passed": ["mean"],   # bool列の平均 ＝ 合格率
    },
)

result = sim.run(n_trials=1_000_000, chunk_size=100_000, seed=42)
print(result.summary())
print(result.value("area", "mean"))   # ≈ 50.0
print(result.to_polars())              # 縦持ちの集計テーブル
```

trial は **必要な入力だけ** を引数に取れます。スカラの返り値は自動でブロードキャストされます。

### frame スタイル — Polars 式

Polars で表現したいパイプライン（遅延最適化や、NumPy では書きにくい操作）向け:

```python
import polars as pl

def trial(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        area=pl.col("w") * pl.col("h"),
    ).with_columns(
        passed=pl.col("area") >= 48.0,
    )
```

## API

### `Simulation(inputs, trial, outputs, quantile_sample_size=200_000, trial_style="auto")`

- `inputs: dict[str, Distribution]` — 入力列名 → 分布。
- `trial` — **array** 関数 `(**入力配列) -> {列名: 配列}`、または **frame** 関数
  `(pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame`。`outputs` に挙げた列をすべて生成する必要があります。
- `outputs: dict[str, list[統計量]]` — 出力列ごとに計算する統計量。
- `quantile_sample_size` — 分位点推定に使うプール標本の予算サイズ。
- `trial_style` — `"auto"`（既定）/ `"array"` / `"frame"`。

`sim.run(n_trials, *, chunk_size=None, seed=0) -> SimulationResult`

- `n_trials` — 総試行回数。
- `chunk_size` — 1チャンクの行数。`None` の場合は既定値（`DEFAULT_CHUNK_SIZE = 250_000`）と `n_trials` の小さい方。
- `seed` — マスターシード。同一シード・同一チャンクサイズなら結果は完全に再現します。

### `SimulationResult`

- `result.value(列, 統計量) -> float` — 単一の統計量を取得。例: `result.value("area", "mean")`。
- `result[列] -> dict[統計量, float]` — 列の全統計量。
- `result.to_dict()` — `{列: {統計量: 値}}`。
- `result.to_polars()` — `column / statistic / value` の縦持ち DataFrame。
- `result.summary()` — 人間が読める複数行サマリ（`repr` も同じ）。
- `result.approximate_columns` — 分位点が近似になった列名の集合。
- そのほか `n_trials` / `n_chunks` / `seed` を保持。

### 統計量

`"count"`、`"sum"`、`"mean"`、`"var"`、`"std"`、`"min"`、`"max"`、`"median"`、
および `("q", 確率)`（例: `("q", 0.95)`）。

### 分布

`Normal(mean, std)`、`Uniform(low, high)`、`LogNormal(mu, sigma)`、
`Triangular(left, mode, right)`、`Exponential(scale)`、`Bernoulli(p)`、
`Constant(value)`。`Distribution.sample(rng, n)` を実装すれば独自分布を追加できます。

## チャンクサイズの指針

ベクトル化版では、チャンクサイズの基準は処理時間ではなく **メモリとスレッド飽和** です。1チャンク ＝ 10万〜100万行程度を目安に、`行数 × 列数` が RAM／キャッシュに収まる範囲で調整してください。`chunk_size` 未指定時は `DEFAULT_CHUNK_SIZE` と `n_trials` の小さい方が使われます。

## 注意点

- array スタイルでは `if width > 5:` のようなスカラ前提の分岐は配列に対して動きません。`np.where(width > 5, a, b)` のように書いてください。算術・比較・NumPy 関数で書く限り、普通の関数のままフル速度で動きます。
- Windows のコンソール（cp932）で `to_polars()` の結果を `print` すると、罫線文字でエンコードエラーになることがあります。ライブラリの不具合ではなく端末の文字コードの問題です。`PYTHONIOENCODING=utf-8` を設定すれば解消します。

## 開発

```bash
uv sync               # 実行用 + 開発用の依存をインストール
uv run pytest         # テスト
uv run ruff check .   # リント
uv run mypy src       # 型チェック（strict）
uv run python examples/rectangle.py
```
