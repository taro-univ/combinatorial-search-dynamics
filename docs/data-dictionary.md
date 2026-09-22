# Data Dictionary

## Schema version

現行schema versionは`2`である。新規書込みはversion 2に限定し、旧version 1は読取り・検証互換を残す。全Parquet tableとZarr rootにversionを保存する。

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
| trial_id | string | no | instance、条件、search seedから生成したID |
| instance_id | string | no | 対象instance |
| experiment_id | string | no | 実験計画 |
| search_method_name | string | no | 探索手法名 |
| search_method_revision | string | yes | 探索実装revision |
| search_seed | int64 | no | 探索乱数seed |
| budget_type | string | no | `candidate_evaluations` |
| budget_limit | int64 | no | 候補評価回数の上限 |
| search_parameters_json | string | no | 解決済み探索パラメータ |
| initial_state_json | string | no | 共通初期状態 |
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
| checkpoint_id | string | no | trial IDと記録順から生成したID |
| trial_id | string | no | 対象trial |
| budget_used | int64 | no | 使用済み候補評価数 |
| checkpoint_index | int64 | no | trial内連番 |
| decision_step | int64 | no | 探索判断の回数 |
| accepted_moves | int64 | no | 採用移動数 |
| rejected_moves | int64 | no | 棄却候補数 |
| action_json | string | yes | 直前候補と採否 |
| state_json | string | no | 外部状態 |
| objective_value | float64 | no | 現在目的値 |
| optimality_gap | float64 | yes | 最適性gap |
| remaining_budget | int64 | no | 残り候補評価数 |
| is_terminal | bool | no | 終端か |
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

Parquetは試行メタデータと表形式の正本であり、Zstandard圧縮を使う。Zarrの`external_state`には現在のbit vectorとboolean maskを保存する。root metadataのsourceは`classical_search`である。

Zarrの`index/checkpoint_id`、`index/trial_id`、`index/budget_used`をParquetの同名列と照合する。version 1では最後の列名が`generated_token_index`である。

## Data layers

- `raw`: 生成器やsolverから得た事実。追記専用を原則とし、既存ファイルは明示的なoverwriteなしに置換しない。
- `interim`: rawから決定的に再生成できる解析前の整形データ。
- `derived`: split、特徴量、状態、遷移、予測、評価など学習・解析依存の成果物。

Phase 2のrawは`data/raw/pilot/`に4表とZarrをまとめたPhase 1の検証可能なsnapshotである。rawの`metrics.parquet`は空schema付き、test評価値は`data/derived/pilot/metrics.parquet`に保存する。`interim`は復元済みtrajectory、`derived`はfeatures、split、状態割当、予測と評価値を持ち、model・report・provenanceは`artifacts/pilot/`へ置く。生成物はDVC cache対象でGit管理しない。MLflowはlocal runを記録する。

`data/derived/pilot/splits.json`はinstance ID→train/validation/testの対応、seed、比率、version、canonical JSONのSHA-256 hashを持つ。同一instanceの全trial/checkpointは同じsplitに属する。状態0はHamming距離0の吸収success状態である。

test metricsは`one_step_nll`（観測次状態確率を最低`1e-12`にclipした負の自然対数のtransition平均）、`one_step_accuracy`（最大確率の次状態との一致率）、`success_brier_score`（次状態が0かという二値事象の二乗誤差平均）。いずれも1 stepで`n_units=transition数`。`evaluated_transitions`、`evaluated_instances`、`evaluated_trials`は件数をmetric_valueにし`n_units=1`。予測JSONに各transitionのinstance ID、trial IDと両checkpoint IDを記録し、instance単位で再集計できる。

## DuckDB views

4つのParquetは同名のDuckDB viewとして直接参照できる。`analysis_checkpoints`は`instances.instance_id = trials.instance_id`、`trials.trial_id = checkpoints.trial_id`でjoinし、checkpointごとの問題条件・trial条件・状態を横断分析する。metricsは多対多化による行数増加を避けるため、このviewには含めない。

## Success-analysis tables

`success_features.parquet`は非終端checkpointごとにB2、M2、M3、目的変数、split、探索手法、trial内重みを持つ。`instance_features.parquet`はsamplingで測る問題特徴を状態特徴から分離する。`success_predictions.parquet`は固定後のtestについてB2、選択済みM2、最終モデルの確率を保存する。3表は`analysis_version` metadataを持ち、rawではなくderivedへ置く。

## Phase 3 optional extension: reference_solutions

ナップサックdatasetでは`reference_solutions.parquet`を同じraw snapshotに要求する。`reference_id`は`instance_id`、solver名/version、全parameter、solve seedから生成する。

| Column | Type | Nullable | Meaning |
|---|---|---:|---|
| schema_version | string | no | 現行`2` |
| reference_id | string | no | 安定した参照結果ID |
| instance_id | string | no | 対象instanceの外部キー |
| task_name | string | no | `knapsack` |
| task_version | string | no | Task定義version |
| solver_name | string | no | `ortools_cp_sat` |
| solver_version | string | no | OR-Tools version |
| solver_status | string | no | OPTIMAL/FEASIBLE/INFEASIBLE/MODEL_INVALID/UNKNOWN/ERROR |
| best_feasible_value | float64 | yes | solverが発見した実行可能目的値 |
| best_bound | float64 | yes | 最大化問題の最適値の上界 |
| optimal_value | float64 | yes | OPTIMALで証明された場合のみ設定 |
| optimality_gap | float64 | yes | solverの上界と実行可能値の相対gap |
| optimality_proven | bool | no | OPTIMALだけtrue |
| timed_out | bool | no | 時間制限による終了の明示分類 |
| runtime_seconds | float64 | no | solver実行時間。mock runtimeとは別 |
| solution_json | string | yes | 0/1選択vectorのcanonical JSON |
| solver_parameters_json | string | no | 完全なsolver条件のcanonical JSON |
| solve_seed | int64 | no | CP-SAT seed |
| error_type | string | yes | 失敗型の短い識別子 |
| created_at | timestamp[us, UTC] | no | 記録時刻。IDと科学的結果には使わない |

Parquet schema metadataには`logical_table=reference_solutions`と`schema_version=1`が入る。Zstandard圧縮、書込後読戻し検証、atomic rename、既定の上書き拒否はPhase 1 APIを再利用する。solverの最大化gapは、実行可能値`z`と上界`b`に対し`max(0, b-z)/max(abs(z), epsilon)`、既定`epsilon=1e-9`。解または上界がなければnull、OPTIMALなら0。checkpointの`optimality_gap`は異なる量で、証明済みoptimal値`o`と現在価値`v`に対し`max(0, o-v)/max(abs(o), epsilon)`である。未証明参照ではnull。両者を混同しない。

ナップサックinstanceの`instance_json`はweights、values、capacity、item_count、generation seed、task名/versionを持つ。checkpointの`state_json`はselected vector、total weight/value、step等を持ち、capacity、actionによる遷移、再計算値を検査する。`objective_value`は現在のtotal value。rawには4必須表、参照表、external-only Zarrを配置し、interimは復元済みtrajectory、derivedは特徴/split/状態/評価値、artifactsはmodel/report/provenanceを持つ。

問題固有のtest metricsは`knapsack_success_rate_trial`（n_units=trial数）、`knapsack_success_rate_instance`（instance数）、`knapsack_final_value_mean`と`knapsack_best_so_far_value_mean`（trial数）、`knapsack_final_absolute_gap_mean`と`knapsack_final_relative_gap_mean`（証明済み参照を持つtrial数）、`knapsack_optimal_reached_budget_mean`（到達trial数）、`knapsack_feasible_checkpoint_rate`（checkpoint数）、`solver_timeout_rate_test`と`solver_optimality_proven_rate_test`（test instance数）である。到達budgetは最適解を最初に記録した時点の候補評価数である。
