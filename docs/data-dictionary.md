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
| created_at | timestamp[us, UTC] | no | 作成時刻 |

### trials

主キーは`trial_id`、外部キーは`instance_id → instances.instance_id`。

| Column | Type | Nullable | Meaning |
|---|---|---:|---|
| schema_version | string | no | schema version |
| trial_id | string | no | instance、条件、sampling seedから生成したID |
| instance_id | string | no | 対象instance |
| experiment_id | string | no | 実験計画 |
| llm_name | string | no | モデル識別子 |
| llm_revision | string | yes | モデルrevision |
| sampling_seed | int64 | no | sampling seed |
| temperature | float64 | no | 生成温度 |
| max_new_tokens | int64 | no | token上限 |
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
| generated_token_index | int64 | no | 生成token位置 |
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
| n_units | int64 | no | 独立評価単位数 |

## Parquet and Zarr correspondence

Parquetは試行メタデータと表形式の正本であり、Zstandard圧縮を使う。Zarrはhidden、attention summary、MLP update、KV summaryの多次元配列を保存し、値と`valid_mask`を分離する。

Zarrの`index/checkpoint_id`、`index/trial_id`、`index/generated_token_index`をParquetの同名列と照合する。配列の保存順やParquetの行番号だけでjoinしない。各観測配列の先頭軸はcheckpointで、index配列の長さと一致する。

## Data layers

- `raw`: 生成器やsolverから得た事実。追記専用を原則とし、既存ファイルは明示的なoverwriteなしに置換しない。
- `interim`: rawから決定的に再生成できる解析前の整形データ。
- `derived`: split、特徴量、状態、遷移、予測、評価など学習・解析依存の成果物。

これらの保存データは将来DVC管理対象とするが、Phase 1ではDVC pipelineを実装しない。

## DuckDB views

4つのParquetは同名のDuckDB viewとして直接参照できる。`analysis_checkpoints`は`instances.instance_id = trials.instance_id`、`trials.trial_id = checkpoints.trial_id`でjoinし、checkpointごとの問題条件・trial条件・状態を横断分析する。metricsは多対多化による行数増加を避けるため、このviewには含めない。
