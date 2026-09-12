# Data Dictionary

## Schema version

現行schema versionは`1`であり、`src/llm_search_dynamics/data/schemas.py`で一元管理する。全Parquet tableは先頭列`schema_version: string non-null`とschema metadataの`schema_version`を持つ。metadataには`logical_table`も保存する。Zarr root attributesにも同じversionを保存する。versionが一致しないデータは読み替えずに拒否し、migrationは将来別機能として追加する。

## Logical tables

nullableと明記した列以外はnon-nullである。列順は以下の順序で固定する。

### instances

主キーは`instance_id`。

| Column | Type | Nullable | Meaning |
|---|---|---:|---|
| schema_version | string | no | schema version |
| instance_id | string | no | canonical問題定義から生成したID |
| task_name | string | no | 問題種 |
| task_version | string | no | 問題定義version |
| problem_size | int64 | no | 問題サイズ |
| difficulty_value | float64 | yes | 問題固有の難易度 |
| generation_seed | int64 | no | instance生成seed |
| instance_json | string | no | canonicalな問題表現 |
| created_at | timestamp[us, UTC] | no | 作成時刻。Phase 2のdummy fixtureではseedから固定 |

### trials

主キーは`trial_id`、外部キーは`instance_id → instances.instance_id`。

| Column | Type | Nullable | Meaning |
|---|---|---:|---|
| schema_version | string | no | schema version |
| trial_id | string | no | instance、条件、sampling seedから生成したID |
| instance_id | string | no | 対象instance |
| experiment_id | string | no | 実験計画 |
| llm_name | string | no | Phase 2では`mock_binary_search`。LLMではない |
| llm_revision | string | yes | Phase 2ではmock generator revision |
| sampling_seed | int64 | no | sampling seed |
| temperature | float64 | no | Phase 2 mockでは探索確率`1 - greedy_probability` |
| max_new_tokens | int64 | no | Phase 2 mockではstep budget |
| terminal_class | string | no | 終端分類 |
| success | bool | no | 最終成功 |
| runtime_seconds | float64 | no | 実行時間 |
| status | string | no | pending/running/completed/failed/interrupted |
| error_type | string | yes | 失敗分類 |

### checkpoints

主キーは`checkpoint_id`、外部キーは`trial_id → trials.trial_id`。同一trial内の`checkpoint_index`も一意でなければならない。

| Column | Type | Nullable | Meaning |
|---|---|---:|---|
| schema_version | string | no | schema version |
| checkpoint_id | string | no | trial IDとtoken位置から生成したID |
| trial_id | string | no | 対象trial |
| generated_token_index | int64 | no | Phase 2 mockでは初期0からのstep位置 |
| checkpoint_index | int64 | no | trial内連番 |
| state_json | string | yes | 外部状態 |
| objective_value | float64 | yes | 現在目的値 |
| optimality_gap | float64 | yes | 最適性gap |
| remaining_budget | int64 | no | 残りtoken数 |
| is_terminal | bool | no | 終端か |
| parse_status | string | no | 状態復元結果 |
| tensor_ref | string | yes | Zarr上の参照 |

### metrics

metricを縦持ちで保存する。仕様上、単一列の主キーは定義しない。`run_id`、`split`、`fold`、`horizon`、`metric_name`の組を分析上の識別に使用する。

| Column | Type | Nullable | Meaning |
|---|---|---:|---|
| schema_version | string | no | schema version |
| run_id | string | no | 実行ID |
| split | string | no | train/validation/test |
| fold | int64 | yes | fold番号 |
| horizon | int64 | yes | 予測先 |
| metric_name | string | no | 指標名 |
| metric_value | float64 | no | 指標値 |
| n_units | int64 | no | Phase 2の予測指標ではtransition数、件数指標では1 |

## Parquet and Zarr correspondence

Parquetは試行メタデータと表形式の正本であり、Zstandard圧縮を使う。Zarrはhidden、attention summary、MLP update、KV summaryの多次元配列を保存し、値と`valid_mask`を分離する。Phase 2では後方互換の追加group `external_state`だけに現在のbit vectorとboolean maskを保存する。rootの`observation_metadata`には`mock`と`external-only`を記録し、内部状態用groupは偽造しない。

Zarrの`index/checkpoint_id`、`index/trial_id`、`index/generated_token_index`をParquetの同名列と照合する。配列の保存順やParquetの行番号だけでjoinしない。各観測配列の先頭軸はcheckpointで、index配列の長さと一致する。

## Data layers

- `raw`: 生成器やsolverから得た事実。追記専用を原則とし、既存ファイルは明示的なoverwriteなしに置換しない。
- `interim`: rawから決定的に再生成できる解析前の整形データ。
- `derived`: split、特徴量、状態、遷移、予測、評価など学習・解析依存の成果物。

Phase 2のrawは`data/raw/pilot/`に4表とZarrをまとめたPhase 1の検証可能なsnapshotである。rawの`metrics.parquet`は空schema付き、test評価値は`data/derived/pilot/metrics.parquet`に保存する。`interim`は復元済みtrajectory、`derived`はfeatures、split、状態割当、予測と評価値を持ち、model・report・provenanceは`artifacts/pilot/`へ置く。生成物はDVC cache対象でGit管理しない。MLflowはlocal runを記録する。

`data/derived/pilot/splits.json`はinstance ID→train/validation/testの対応、seed、比率、version、canonical JSONのSHA-256 hashを持つ。同一instanceの全trial/checkpointは同じsplitに属する。状態0はHamming距離0の吸収success状態である。

test metricsは`one_step_nll`（観測次状態確率を最低`1e-12`にclipした負の自然対数のtransition平均）、`one_step_accuracy`（最大確率の次状態との一致率）、`success_brier_score`（次状態が0かという二値事象の二乗誤差平均）。いずれも1 stepで`n_units=transition数`。`evaluated_transitions`、`evaluated_instances`、`evaluated_trials`は件数をmetric_valueにし`n_units=1`。予測JSONに各transitionのinstance ID、trial IDと両checkpoint IDを記録し、instance単位で再集計できる。

## DuckDB views

4つのParquetは同名のDuckDB viewとして直接参照できる。`analysis_checkpoints`は`instances.instance_id = trials.instance_id`、`trials.trial_id = checkpoints.trial_id`でjoinし、checkpointごとの問題条件・trial条件・状態を横断分析する。metricsは多対多化による行数増加を避けるため、このviewには含めない。
