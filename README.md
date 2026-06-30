# polars_mc

[Polars](https://pola.rs/) を基盤にしたベクトル化モンテカルロ・シミュレーション。

利用者が書くのは **1試行（1行＝1試行）だけ**。再現可能な乱数サンプリング・チャンク分割・並列評価・チャンク横断の集計合成は、すべて実行基盤が隠蔽します。モデル作成者はチャンク・スレッド・乱数シードを一切意識する必要がありません。

## 特長

- **書くのは1試行だけ** — 普通の Python 関数（NumPy）として書け、ブロードキャストで全試行ぶんが一気にベクトル化実行される。
- **完全な再現性** — 乱数はチャンクごとに独立シード。同一シードなら、チャンクサイズ・バックエンド・ワーカー数に関係なくビット単位で同一結果。
- **メモリ有界** — 何億試行でも、各チャンクは小さな集計だけを返すので一定メモリで回る。
- **2つの並列バックエンド** — 軽い計算は Polars 内蔵スレッド、重いCPU処理（Shapely 等）はプロセス並列。
- **正確な集計合成** — 平均・分散は数値的に安定な Chan のアルゴリズムでチャンク横断合成。

## インストール

[uv](https://docs.astral.sh/uv/) でこのリポジトリを使う場合:

```bash
uv sync          # 実行用 + 開発用の依存をインストール
```

別プロジェクトから依存に加える場合（ローカルパス例）:

```bash
uv add path/to/polars-mc
```

ランタイム依存は `polars` と `numpy` のみです。

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
        "passed": ["mean"],   # bool列の平均 ＝ 合格率
    },
)

# 3) 実行
result = sim.run(n_trials=1_000_000, seed=42)

# 4) 結果を読む
print(result.summary())
print(result.value("area", "mean"))      # ≈ 50.0
print(result.value("passed", "mean"))    # ≈ 0.92（合格率）
```

`result.summary()` の出力例:

```
Monte Carlo simulation: 1,000,000 trials in 4 chunk(s), seed=42
  area (quantiles approximate):
    mean     = 49.9983
    std      = 1.4139
    min      = 43.669
    max      = 56.8783
    q0.95    = 52.3414
  passed:
    mean     = 0.922322
```

## trial の書き方

trial は2つのスタイルで書けます。エンジンが関数のシグネチャから自動判定します（`trial_style=` で明示も可）。引数名が入力名と一致すれば **array**、`(df)` のように1つの frame を取れば **frame** と判定されます。

### array スタイル — 普通の関数（推奨）

入力は NumPy 配列として（チャンクぶんをまとめて）渡され、`{出力列名: 配列}` を返します。NumPy がブロードキャストするので、「1試行ぶんを書いたつもりの普通のコード」がそのまま全試行ぶん完全にベクトル化されます。

```python
def trial(width, height):
    area = width * height
    return {"area": area, "passed": area >= 48.0}
```

ルール:

- 引数は **必要な入力名だけ** 書けばよい（使わない入力は省略可）。
- 返り値は `dict[str, 配列]`。長さ `n` の配列、またはスカラ（自動でブロードキャスト）。
- 入力列もそのまま集計対象にできる（`outputs` に入力名を指定可）。
- スカラ前提の `if` は使えない → `np.where` / `np.select` / `np.clip` などで書く。

```python
# 条件分岐は np.where で
grade = np.where(area >= 50.0, 1.0, 2.0)
```

### frame スタイル — Polars 式

Polars の遅延最適化を使いたい、または NumPy では書きにくいパイプライン向け:

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

`Distribution` を継承し、`sample(rng, n)` で長さ `n` の NumPy 配列を返すだけです。

```python
import numpy as np
from polars_mc import Distribution, Simulation

class Categorical(Distribution):
    """重み付きでカテゴリ（整数コード）を選ぶ離散分布。"""
    def __init__(self, weights):
        w = np.asarray(weights, dtype=float)
        self.p = w / w.sum()
    def sample(self, rng: np.random.Generator, n: int):
        return rng.choice(len(self.p), size=n, p=self.p)

sim = Simulation(
    inputs={"grade": Categorical([0.5, 0.3, 0.2])},
    trial=lambda grade: {"grade": grade},
    outputs={"grade": ["mean"]},
)
```

`rng` はチャンクごとに独立シード済みの `numpy.random.Generator` です。これを使えば再現性が保たれます。

## 出力と統計量

`outputs` は `{列名: [統計量, ...]}` です。列は trial が生成した列、または入力列を指定できます。

| 統計量 | 内容 |
|---|---|
| `"count"` | 件数 |
| `"sum"` | 合計 |
| `"mean"` | 平均 |
| `"var"` | 分散（不偏、ddof=1） |
| `"std"` | 標準偏差（`var` の平方根） |
| `"min"` / `"max"` | 最小 / 最大 |
| `"median"` | 中央値（`("q", 0.5)` と同じ） |
| `("q", p)` | `p` 分位点（例: `("q", 0.95)`） |

bool 列の `"mean"` は **True の割合**（合格率・確率）になります。

### 分位点の厳密性について

平均・分散・min・max・count は **チャンク横断で厳密に合成**できます。一方 **分位点は加法的に合成できない** ため、列ごとに一様サブサンプルを上限つき（`quantile_sample_size`、既定20万）で保持し、その標本から推定します。

- 総試行数が `quantile_sample_size` 以内 → 全データ保持で **厳密**。
- 予算を超える → **近似**。その列名が `result.approximate_columns` に入る。

厳密にしたい場合は `quantile_sample_size` を総試行数以上に設定してください（その分メモリを使います）。

## 結果の扱い（`SimulationResult`）

```python
result = sim.run(1_000_000, seed=42)

result.value("area", "mean")     # 単一の統計量 -> float
result["area"]                    # その列の全統計量 -> {"mean": ..., "std": ...}
result.to_dict()                  # {列: {統計量: 値}}
result.to_polars()                # column / statistic / value の縦持ち DataFrame
print(result.summary())           # 人が読む複数行サマリ（repr も同じ）

result.approximate_columns        # 分位点が近似になった列名の集合
result.n_trials, result.n_chunks, result.seed
```

`to_polars()` は後段の分析・保存に便利です（例: `result.to_polars().write_csv("out.csv")`）。

## 実行オプション（`sim.run`）

```python
sim.run(
    n_trials,            # 総試行回数
    *,
    chunk_size=None,     # 1チャンクの行数。None なら min(n_trials, 250_000)
    seed=0,              # マスターシード
    backend="sequential",# "sequential" | "processes"
    n_workers=None,      # processes 時のワーカー数（既定: 全CPU）
)
```

### バックエンドの選び方

| trial の中身 | backend | 理由 |
|---|---|---|
| 算術・補間・VLOOKUP など軽い処理 | `"sequential"`（既定） | Polars 内蔵スレッドで十分速い |
| Shapely 等の重いCPU処理 | `"processes"` | チャンクが独立CPU作業。コア数にほぼ比例して高速化 |

`backend="processes"` は **ピクル可能な trial** が必要です（モジュールトップレベル関数にする。ラムダ・局所関数は不可）。スクリプトのエントリは必ず `if __name__ == "__main__":` でガードしてください。再現性はバックエンドに依存しません（同一シードなら逐次と完全一致）。

```python
result = sim.run(1_000_000, seed=42, backend="processes", n_workers=8)
```

## サンプル集

| ファイル | 内容 |
|---|---|
| [`examples/rectangle.py`](examples/rectangle.py) | 基本。歩留まり計算（array スタイル） |
| [`examples/lookup.py`](examples/lookup.py) | テーブル参照／補間（VLOOKUP 的処理）。`np.interp` / `np.searchsorted` / ファンシーインデックス |
| [`examples/shapely_intersection.py`](examples/shapely_intersection.py) | Shapely の幾何交差。`backend="processes"` の効果も計測 |

実行例:

```bash
uv run python examples/rectangle.py
uv run python examples/lookup.py
uv run --with shapely python examples/shapely_intersection.py
```

## チャンクサイズの指針

ベクトル化版では、チャンクサイズの基準は処理時間ではなく **メモリとスレッド飽和** です。1チャンク ＝ 10万〜100万行程度を目安に、`行数 × 列数` が RAM／キャッシュに収まる範囲で調整してください。`chunk_size` 未指定時は `DEFAULT_CHUNK_SIZE`（250,000）と `n_trials` の小さい方が使われます。

`backend="processes"` では、チャンク数がワーカー数より十分多いほど負荷が均等に分散します。チャンクが大きすぎる（＝チャンク数が少ない）と並列度が出ません。

## 注意点

- array スタイルでは `if width > 5:` のようなスカラ前提の分岐は配列に対して動きません。`np.where(width > 5, a, b)` のように書いてください。算術・比較・NumPy 関数で書く限り、普通の関数のままフル速度で動きます。
- 試行ごとに発散する制御フローや、スカラ専用の外部ソルバを1試行ごとに呼ぶような処理は、ベクトル化の守備範囲外です（Shapely のように配列対応APIがあるものは別）。
- Windows のコンソール（cp932）で `to_polars()` の結果を `print` すると、罫線文字でエンコードエラーになることがあります。ライブラリの不具合ではなく端末の文字コードの問題です。`PYTHONIOENCODING=utf-8` を設定すれば解消します。

## 設計

```
入力仕様層   distributions.py   Normal / Uniform / LogNormal / ...
モデル層     （利用者のtrial関数）  1行＝1試行          ← 利用者が書くのはここだけ
RNG層        rng.py             seed → チャンク別の独立な乱数生成器
チャンク層   chunk.py           入力サンプリング → trial適用 → 部分集計
集計層       aggregate.py       加法的モーメント合成 + 分位点サブサンプリング
実行層       engine.py          チャンク計画・並列実行・合成
結果層       result.py          SimulationResult（summary / to_dict / to_polars）
```

チャンクをまたいで厳密に合成できるのは **加法的な統計量** だけです。そのためエンジンは列ごとに count / mean / M2（偏差平方和）/ min / max を保持し、数値的に安定な Chan の並列アルゴリズムで合成します。分位点だけは合成不能なのでサブサンプル推定にしています（上記「分位点の厳密性について」を参照）。

並列化は、既定では Polars 内蔵スレッドプール（チャンク内の式評価を並列化）、`backend="processes"` ではチャンク単位のプロセス分散に委譲します。チャンクループは逐次なのでメモリが有界に保たれ、独立シードにより結果は完全に再現します。

## 開発

```bash
uv sync               # 実行用 + 開発用の依存をインストール
uv run pytest         # テスト（39件）
uv run ruff check .   # リント
uv run mypy src       # 型チェック（strict）
```
