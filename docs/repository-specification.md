# Combinatorial Search Dynamics リポジトリ仕様書

| 項目 | 内容 |
|---|---|
| 文書状態 | 初期構築仕様 |
| 文書版 | 0.1.0 |
| 想定リポジトリ名 | `combinatorial-search-dynamics` |
| Pythonパッケージ名 | `combinatorial_search_dynamics` |
| 想定Python | 3.11 |
| 対象 | M1後期の研究実験・解析基盤 |

## 1. 目的

本リポジトリは、LLMによる組合せ最適化の探索過程を、問題空間上の確率運動と潜在記憶の結合系として記録・解析・モデル化するための研究基盤を提供する。

本仕様が定めるのは研究結果そのものではなく、次を再現可能に実行するための共通基盤である。

1. 問題インスタンスの生成と正解・基準値の計算
2. LLMによる探索軌道の生成
3. 問題空間およびLLM内部の観測量の記録
4. 状態表現と力学モデルの学習
5. ハイパーパラメータ探索
6. held-outデータによる評価
7. 結果、設定、データ、コードの追跡
8. 複数の問題・モデル・難易度間の比較

## 2. 設計原則

### 2.1 関心の分離

以下を独立したモジュールとして分離する。

- 問題の定義
- LLMによる生成
- 観測量の抽出
- データの整形
- 状態表現
- 動力学モデル
- 評価指標
- 可視化・報告

問題や状態表現を変更しても、それ以外の処理を書き換えずに実験できることを目標とする。

### 2.2 設定優先

研究条件をPythonコードへ直接埋め込まない。Hydraの設定ファイルを実験条件の正本とし、コードは設定を受け取って動作する。

### 2.3 不変な生データ

LLM生成直後のデータを`raw`として保存し、後処理で上書きしない。修正された抽出・解析処理は、`interim`または`derived`を再生成する。

### 2.4 分割後の学習

標準化、次元削減、クラスタリング、状態モデル、ハイパーパラメータ選択は、すべて訓練側だけで学習する。testデータは最終評価まで使用しない。

### 2.5 明示的な識別子

ParquetとZarrの対応を行番号や保存順に依存させない。すべてのデータを明示的なIDで結合する。

### 2.6 失敗も成果物として残す

生成失敗、構文解析失敗、solver timeout、OOM、途中終了などを削除せず、状態と理由を記録する。

## 3. 採用技術と責務

| 技術 | 責務 | 責務外 |
|---|---|---|
| Git | コード、設定、文書、小さな集計結果の版管理 | 大容量データ、モデル本体 |
| uv | Python、直接・間接依存関係、仮想環境の固定 | 実験条件の管理 |
| Hydra | 実験設定の合成、上書き、実行時設定の保存 | 評価結果の履歴管理 |
| Optuna | inner validation内のハイパーパラメータ探索 | 問題や評価条件の恣意的選択 |
| MLflow | run、parameter、metric、artifactの追跡 | 生データの版管理 |
| DVC | データ、特徴量、大容量成果物、pipelineの版管理 | Python依存関係の固定 |
| Parquet | 表形式データの永続化 | 高次元テンソルの主保存 |
| DuckDB | 複数Parquetの検索、結合、集計 | データの唯一の保存先 |
| Zarr | hidden、attention、KV等の多次元配列の保存 | 試行メタデータの主保存 |
| OR-Tools | 正解、境界、実行可能性、基準解の計算 | LLM側の探索ロジック |

補助的な開発依存として、`pytest`、`ruff`、必要に応じて`Typer`を使用する。CLIは当初Python moduleとして実装してもよいが、外部向けコマンド名は本仕様に従う。

## 4. リポジトリ構成

```text
combinatorial-search-dynamics/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .dvcignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── configs/
│   ├── config.yaml
│   ├── experiment/
│   ├── task/
│   ├── llm/
│   ├── generation/
│   ├── observation/
│   ├── state_model/
│   ├── dynamics/
│   ├── evaluation/
│   ├── storage/
│   └── search_space/
├── src/
│   └── combinatorial_search_dynamics/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── identifiers.py
│       ├── provenance.py
│       ├── tasks/
│       │   ├── base.py
│       │   └── registry.py
│       ├── solvers/
│       │   ├── base.py
│       │   └── ortools_solver.py
│       ├── generation/
│       │   ├── runner.py
│       │   ├── parser.py
│       │   └── stopping.py
│       ├── observation/
│       │   ├── base.py
│       │   ├── external.py
│       │   └── internal.py
│       ├── data/
│       │   ├── schemas.py
│       │   ├── parquet.py
│       │   ├── zarr.py
│       │   ├── validation.py
│       │   └── splits.py
│       ├── features/
│       │   ├── external.py
│       │   ├── history.py
│       │   └── internal.py
│       ├── state_models/
│       │   ├── base.py
│       │   ├── preprocessing.py
│       │   └── registry.py
│       ├── dynamics/
│       │   ├── base.py
│       │   ├── transition.py
│       │   ├── absorbing.py
│       │   └── history.py
│       ├── tuning/
│       │   ├── objective.py
│       │   ├── search_space.py
│       │   └── selection.py
│       ├── evaluation/
│       │   ├── prediction.py
│       │   ├── markov.py
│       │   ├── uncertainty.py
│       │   └── baselines.py
│       ├── tracking/
│       │   ├── mlflow.py
│       │   └── artifacts.py
│       └── reporting/
│           ├── figures.py
│           └── tables.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── scripts/
│   ├── smoke_test.ps1
│   └── reproduce_pilot.ps1
├── data/
│   ├── raw/
│   ├── interim/
│   ├── derived/
│   └── external/
├── artifacts/
│   ├── studies/
│   ├── models/
│   ├── figures/
│   └── reports/
├── mlruns/
├── notebooks/
│   └── exploratory/
└── docs/
    ├── repository-specification.md
    ├── architecture.md
    ├── data-dictionary.md
    ├── experiment-protocol.md
    └── adr/
```

### 4.1 配置規則

- 再利用される処理は`src/`へ置く。
- `scripts/`は薄い実行ラッパーに限定する。
- notebookは探索専用とし、本実験の唯一の実装にしない。
- notebookで確定した処理は`src/`へ移す。
- Git管理する図は論文・発表で使用する確定版に限定する。
- `data/`、`artifacts/`、`mlruns/`の大容量ファイルは原則Git管理しない。

## 5. パッケージ境界

### 5.1 Taskインターフェース

各組合せ最適化問題は共通インターフェースを実装する。

```python
class TaskProtocol(Protocol):
    def generate_instance(self, seed: int, **params): ...
    def initial_state(self, instance): ...
    def legal_actions(self, state): ...
    def apply_action(self, state, action): ...
    def is_feasible(self, state) -> bool: ...
    def is_terminal(self, state) -> bool: ...
    def objective(self, state) -> float: ...
    def serialize_state(self, state) -> dict: ...
    def parse_action(self, text: str): ...
```

距離、難易度、最適値は問題によって定義できない場合があるため、必須インターフェースから分離する。

### 5.2 Solverインターフェース

OR-Tools solverは次を返す。

- solver status
- 最良実行可能値
- best bound
- optimality gap
- 実行時間
- timeoutの有無
- 得られた解
- OR-Toolsのversionと主要設定

最適性が証明されていない場合、推定値を`optimal_value`として保存しない。

### 5.3 Observationインターフェース

観測量は次の3群を明確に区別する。

- `external`: 盤面、目的関数、操作、制約違反など
- `history`: 過去状態、滞在時間、改善方向、反復など
- `internal`: hidden、attention、MLP更新量、KV要約など

各観測量には名前、shape、dtype、取得時点、欠損条件を定義する。

### 5.4 State modelインターフェース

状態表現モデルは最低限、次を実装する。

```python
fit(train_data)
transform(data)
fit_transform(train_data)
save(path)
load(path)
```

標準化器、次元削減器、クラスタリング器を含む場合、それぞれを訓練データだけでfitする。

### 5.5 Dynamics modelインターフェース

```python
fit(train_trajectories)
predict_distribution(state, horizon)
score(test_trajectories)
save(path)
load(path)
```

Markovモデルに限定せず、履歴付きモデルや潜在記憶モデルを追加可能にする。

## 6. 識別子

識別子は文字列として保存し、生成規則を一元管理する。

| ID | 意味 | 推奨生成法 |
|---|---|---|
| `experiment_id` | 一つの実験計画 | 人間可読名＋短いhash |
| `run_id` | 一回の実行 | MLflow run ID |
| `instance_id` | 問題インスタンス | 問題定義のcanonical JSON hash |
| `trial_id` | 一回のLLM探索 | instance、条件、seedのhash |
| `checkpoint_id` | 軌道上の観測点 | trial ID＋生成token位置 |
| `artifact_id` | 配列・モデル等 | 内容hashまたはrun ID＋名称 |

同じ問題内容からは、保存順に関係なく同じ`instance_id`が得られることをテストする。

## 7. データレイヤ

### 7.1 Raw

生成器またはsolverから得た事実を保存する。原則として追記専用とする。

- 問題インスタンス
- prompt
- 生成token列
- sampling条件
- 生成中に取得した外部・内部観測
- solver出力
- 標準出力、エラー、終了状態

### 7.2 Interim

rawから決定的に再生成できる整形データを保存する。

- 復元された盤面軌道
- 構文解析済み操作
- checkpoint対応表
- 検証済み終端ラベル
- 正規化前の観測特徴

### 7.3 Derived

学習や解析に依存する成果物を保存する。

- データ分割
- 標準化済み特徴
- 低次元座標
- 離散状態
- 遷移行列
- committor
- 予測値
- 評価値

Derivedデータには、生成元データのDVC revision、設定hash、fit対象splitを付与する。

## 8. Parquet仕様

初期版では少なくとも次の論理テーブルを使用する。

### 8.1 `instances.parquet`

| 列 | 型 | 説明 |
|---|---|---|
| `instance_id` | string | 主キー |
| `task_name` | string | 問題種 |
| `task_version` | string | 問題定義版 |
| `problem_size` | int64 | 問題サイズ |
| `difficulty_value` | float64 nullable | 難易度 |
| `generation_seed` | int64 | 生成seed |
| `instance_json` | string | canonicalな問題表現 |
| `created_at` | timestamp UTC | 作成日時 |

### 8.2 `trials.parquet`

| 列 | 型 | 説明 |
|---|---|---|
| `trial_id` | string | 主キー |
| `instance_id` | string | 外部キー |
| `experiment_id` | string | 実験計画 |
| `llm_name` | string | モデル識別子 |
| `llm_revision` | string nullable | モデルrevision |
| `sampling_seed` | int64 | 生成seed |
| `temperature` | float64 | 温度 |
| `max_new_tokens` | int64 | token上限 |
| `terminal_class` | string | 終端分類 |
| `success` | bool | 最終成功 |
| `runtime_seconds` | float64 | 実行時間 |
| `status` | string | completed/failed等 |
| `error_type` | string nullable | 失敗分類 |

### 8.3 `checkpoints.parquet`

| 列 | 型 | 説明 |
|---|---|---|
| `checkpoint_id` | string | 主キー |
| `trial_id` | string | 外部キー |
| `generated_token_index` | int64 | 生成位置 |
| `checkpoint_index` | int64 | 試行内連番 |
| `state_json` | string nullable | 外部状態 |
| `objective_value` | float64 nullable | 現在目的値 |
| `optimality_gap` | float64 nullable | 最適性gap |
| `remaining_budget` | int64 | 残りtoken |
| `is_terminal` | bool | 終端か |
| `parse_status` | string | 盤面復元状態 |
| `tensor_ref` | string nullable | Zarr上の参照 |

### 8.4 `metrics.parquet`

縦持ち形式を基本とする。

| 列 | 型 | 説明 |
|---|---|---|
| `run_id` | string | MLflow run |
| `split` | string | train/validation/test |
| `fold` | int64 nullable | fold番号 |
| `horizon` | int64 nullable | 予測先 |
| `metric_name` | string | 指標名 |
| `metric_value` | float64 | 指標値 |
| `n_units` | int64 | 独立評価単位数 |

ParquetはZstandard圧縮を既定とする。スキーマ変更時には`schema_version`を更新し、既存ファイルを黙って読み替えない。

## 9. Zarr仕様

Zarrは多次元配列だけに使用し、次のようなgroup構成を基本とする。

```text
observations.zarr/
├── hidden/
│   ├── values
│   └── valid_mask
├── attention_summary/
│   ├── values
│   └── valid_mask
├── mlp_update/
│   ├── values
│   └── valid_mask
├── kv_summary/
│   ├── values
│   └── valid_mask
└── index/
    ├── trial_id
    ├── checkpoint_id
    └── generated_token_index
```

### 9.1 配列規則

- shapeと軸名をZarr attributesへ保存する。
- dtypeをattributesとデータ辞書へ記録する。
- 欠損をゼロ埋めだけで表現せず`valid_mask`を持つ。
- chunkは主な読出単位であるtrialまたはcheckpointを先頭にする。
- 生観測と加工済み要約を同一pathへ上書きしない。
- 保存時にmodel revision、tokenizer revision、観測コード版を付与する。

Zarr format versionとcompressorは、実装時の依存互換性試験後にADRで固定する。初期候補はZarr v3＋Zstandardとするが、互換性に問題がある場合はv2を採用する。

## 10. DuckDB利用方針

DuckDBはParquetに対する分析・検査層として使用する。

- Parquetを正本とする。
- 初期段階では永続DuckDBファイルを正本にしない。
- 再利用するview定義はSQLファイルまたはPythonコードとしてGit管理する。
- 集計結果は必要に応じてParquetへ書き戻す。
- Zarr内の大配列はDuckDBへ複製せず、参照IDだけを扱う。

代表的なviewとして、trial、instance、checkpoint、metricを結合した`analysis_checkpoints`を提供する。

## 11. Hydra設定仕様

`configs/config.yaml`はdefault compositionだけを担う。

```yaml
defaults:
  - experiment: pilot
  - task: placeholder
  - llm: placeholder
  - generation: default
  - observation: external_only
  - state_model: baseline
  - dynamics: markov
  - evaluation: default
  - storage: local
  - _self_
```

### 11.1 設定分類

| 分類 | 例 | Optunaによる変更 |
|---|---|---|
| 科学的実験条件 | 問題種、サイズ、難易度、LLM、温度 | 原則不可 |
| 収集条件 | 試行数、token上限、記録間隔 | 原則不可 |
| モデル選択対象 | 潜在次元、状態数、lag、正則化 | 可 |
| 学習パラメータ | 遷移確率、係数 | データから推定 |
| 環境設定 | path、device、worker数 | 性能比較から除外 |

各実行では解決済みの完全なHydra configをartifactとして保存する。

### 11.2 秘密情報

- API key、token、passwordをYAMLへ書かない。
- 秘密情報は環境変数またはGit管理外の`.env`から取得する。
- `.env.example`には変数名だけを記載する。

## 12. Optuna仕様

### 12.1 適用範囲

Optunaは状態表現・力学モデルのハイパーパラメータだけを探索する。

例：

- 潜在次元
- 離散状態数
- lag
- 履歴長
- 正則化強度
- モデル構造

問題難易度、問題サイズ、モデル規模を「良い結果になるように」選択しない。

### 12.2 データ境界

- trialはouter-train内のinner splitだけを使用する。
- outer-testの結果をOptunaへ返さない。
- preprocessingも各inner-trainでfitする。
- study終了後に選択規則で一つの構成を固定する。
- 固定後のtest評価は原則一度だけ行う。

### 12.3 永続化

- 初期の単一プロセス実行はSQLiteを許容する。
- 並列・複数マシン実行へ移行する場合はPostgreSQL等へ移す。
- `study_name`に実験名、データ版、目的関数版を含める。
- 各Optuna trialをMLflowのnested runとして記録する。

## 13. MLflow仕様

### 13.1 Run階層

```text
experiment
└── outer-fold run
    ├── Optuna trial run
    └── final-fit/evaluation run
```

### 13.2 必須タグ

- `git_commit`
- `dvc_revision`
- `config_hash`
- `schema_version`
- `task_name`
- `task_version`
- `llm_name`
- `llm_revision`
- `split_version`
- `code_status`：clean/dirty

### 13.3 必須artifact

- 解決済みHydra config
- 評価単位ごとのmetric Parquet
- 集計表
- 図
- 学習済みstate/dynamics model
- split定義
- エラー概要

MLflow UI上の値だけを最終記録とせず、論文用集計を再生成できる粒度のParquetも保存する。

## 14. DVC pipeline

初期pipelineは次を標準とする。

```text
generate_instances
       ↓
solve_references
       ↓
collect_trajectories
       ↓
prepare_trajectories
       ↓
extract_features
       ↓
make_splits
       ↓
tune_model
       ↓
fit_model
       ↓
evaluate_model
       ↓
build_report
```

### 14.1 再計算境界

- 問題生成条件の変更：原則全段を再計算
- 観測抽出方法の変更：`extract_features`以降を再計算
- モデル設定の変更：`tune_model`または`fit_model`以降を再計算
- 図の体裁だけの変更：`build_report`だけを再計算

### 14.2 Remote

DVC remoteは初期構築の必須条件としない。ただし、本実験開始前にはバックアップ可能なremoteを設定する。remote固有のcredentialはGitへ保存しない。

## 15. CLI仕様

利用者向けの操作を以下へ統一する。

```text
csd generate-instances
csd solve-references
csd collect
csd prepare
csd extract
csd split
csd tune
csd fit
csd evaluate
csd report
csd validate-data
csd reproduce-pilot
```

各コマンドは以下を満たす。

- Hydra overrideを受け取れる。
- 成功時は終了コード0を返す。
- 失敗時は非0を返し、構造化されたエラーを残す。
- 入力と出力をログに記録する。
- 同じ入力を再実行してもrawデータを暗黙に破壊しない。

## 16. 実行メタデータと再現性

すべての実験で次を記録する。

- Git commitとdirty状態
- DVC revision
- `uv.lock`のhash
- Python version
- OS
- CPU/GPU
- CUDA、PyTorch、Transformers version
- OR-Tools version
- LLMとtokenizerのrevision
- 完全な設定
- seed
- 開始・終了時刻

seedは用途別に分ける。

```yaml
seed:
  instance_generation: 100
  llm_sampling: 200
  data_split: 300
  representation: 400
  dynamics: 500
  optuna: 600
  bootstrap: 700
```

GPU上の完全なbit一致を常に保証するとは限らない。その場合も、再現性モード、使用ハードウェア、非決定的処理の有無を記録する。

## 17. データ分割の実装要件

- splitの基本単位は`instance_id`とする。
- 同一instanceのcheckpointを複数splitへ分散させない。
- 同一trialを複数splitへ分散させない。
- split一覧をParquetまたはJSONとして保存する。
- split生成後は内容hashを付与する。
- test splitの存在を前処理・探索コードから隠せるAPIを用意する。
- 時系列予測では未来側の情報を現在の特徴に混入させない。

## 18. テスト戦略

### 18.1 Unit test

- 盤面操作の合法性
- serialize/deserializeの往復
- IDの決定性
- 遷移行列の非負性と行和1
- 吸収状態の構造
- horizonと行列累乗の対応
- metricの既知例
- config validation

### 18.2 Integration test

小規模な擬似問題とmock生成器により、次をCPUだけで通す。

```text
instance生成 → 軌道保存 → 前処理 → fit → evaluate → report
```

### 18.3 Regression test

固定fixtureに対して、次が許容誤差内で一致することを確認する。

- 遷移行列
- 状態割当
- 主要評価指標
- 生成される表のschema

### 18.4 Smoke test

本物のモデルを使うGPU smoke testはCIの必須項目にせず、ローカルまたは計算機上で明示的に実行する。

## 19. コード品質とCI

Pull Requestまたはmainへの反映前に、最低限次を実行する。

```text
ruff check
ruff format --check
pytest tests/unit tests/integration
```

CIでは大容量データ、外部API、GPU、DVC remoteを必要としない。fixtureは小さく、Git管理可能なものに限定する。

## 20. 文書とADR

### 20.1 README

READMEには以下だけを簡潔に記載する。

- 研究目的
- セットアップ
- pilot実行方法
- ディレクトリ構成
- データ取得方法
- 詳細文書へのリンク

### 20.2 ADR形式

```text
# ADR-NNN タイトル

## Status
Proposed / Accepted / Superseded

## Context
何を決める必要があるか

## Decision
何を採用するか

## Alternatives
検討した代替案

## Consequences
利点、制約、将来の変更条件
```

初期ADRとして以下を作成する。

- ADR-001：uvとPython 3.11
- ADR-002：src layout
- ADR-003：Hydraを設定の正本とする
- ADR-004：ParquetとZarrの責務分離
- ADR-005：DuckDBを問合せ層として使用する
- ADR-006：Git・DVC・MLflowの責務分離
- ADR-007：Optunaの適用範囲
- ADR-008：OR-Tools solverの抽象化

## 21. Git管理方針

### 21.1 Gitへ含める

- `src/`
- `tests/`
- `configs/`
- `docs/`
- `scripts/`
- `pyproject.toml`
- `uv.lock`
- `dvc.yaml`
- `dvc.lock`
- 小さなfixture
- 確定した小容量の表・図

### 21.2 Gitへ含めない

- `.env`
- model weights
- Hugging Face cache
- 生の生成データ
- Zarr store
- 大規模Parquet
- Optuna SQLite実体
- ローカルMLflow DBとartifact
- GPU dump
- 一時ログ

DVC metafileはGit管理し、実データはDVC cache/remoteで管理する。

## 22. エラーと中断復帰

- 各trialの状態を`pending/running/completed/failed/interrupted`で管理する。
- 完了済みtrialは、明示的な`--force`なしに再生成しない。
- 部分的に書かれたParquet/Zarrをcompletedとして扱わない。
- 一時領域へ書き、検証後に正式pathへ確定する。
- OOM、timeout、parse failure、solver failureを区別する。
- 中断後は未完了trialだけを再開可能にする。

## 23. 初期構築の段階

### Phase 0：骨格

- Git初期化
- uv project作成
- src layout作成
- Hydra設定読込み
- Ruff、pytest
- CI
- ADR雛形

### Phase 1：保存基盤

- ID生成
- Parquet I/O
- Zarr I/O
- DuckDB view
- schema version
- データ検証

### Phase 2：最小pipeline

- ダミー問題
- ダミー生成器
- 単純な状態表現
- 単純な遷移モデル
- 評価とMLflow記録
- DVC pipeline

### Phase 3：OR-Toolsと実問題

- 最初の問題adapter
- OR-Tools solver
- feasible check
- optimum/bound/gap記録

### Phase 4：LLM接続

- モデルadapter
- prompt versioning
- sampling記録
- checkpoint観測
- 中断復帰

### Phase 5：探索と本評価

- Optuna inner CV
- nested MLflow runs
- 構成固定
- outer-test
- report生成

## 24. Pilotの完了条件

初期リポジトリは、次を満たした時点で完成とする。

1. `uv sync`だけで環境を再構築できる。
2. 一つのコマンドで小規模pipelineを完走できる。
3. 問題、trial、checkpointが明示的IDで追跡できる。
4. 表をParquet、テンソルをZarrへ保存できる。
5. DuckDBからtrialとcheckpointを横断検索できる。
6. OR-Toolsの解・bound・statusを保存できる。
7. Optuna trialがMLflowへ記録される。
8. DVCが入力変更に応じて必要なstageだけを再実行する。
9. testデータがOptunaおよび前処理のfitに使われない。
10. Git commit、DVC revision、config、seedからrunを特定できる。
11. CPU integration testがCIで成功する。
12. 別ディレクトリでpilot結果を再生成できる。

## 25. 初期既定値と未確定事項

### 25.1 初期既定値

| 項目 | 既定値 |
|---|---|
| リポジトリ名 | `combinatorial-search-dynamics` |
| package | `combinatorial_search_dynamics` |
| Python | 3.11 |
| layout | `src` layout |
| 公開範囲 | private |
| 設定の正本 | Hydra |
| 表形式 | Parquet＋Zstandard |
| テンソル | Zarr |
| Query | DuckDB |
| HPO | Optuna |
| tracking | local MLflow |
| data version | DVC |
| solver | OR-Tools adapter |
| test | pytest |
| lint/format | Ruff |

### 25.2 リポジトリ作成を妨げない未確定事項

- 最初の実問題
- DVC remoteの接続先
- Gitホスティング先と最終的な公開範囲
- Zarr v2/v3の最終選択
- Zarr chunkとcompressorの詳細
- MLflowの将来的なサーバー化
- Optunaの将来的なPostgreSQL化
- CLIにTyperを使うか
- GPU CIの有無

これらはadapterと設定で差し替えられるようにし、決定時にADRを追加する。

## 26. 非目標

初期構築では以下を実装しない。

- すべての組合せ最適化問題への対応
- 分散クラスタ実行
- 本番Webサービス
- 独自experiment tracking UI
- 独自データベース
- LLM内部への因果介入
- 論文用の全解析
- 状態表現や力学モデルの早期な一本化

初期段階の目的は、研究仮説が変化してもデータ収集、状態表現、モデル、評価を交換できる再現可能な最小基盤を完成させることである。

