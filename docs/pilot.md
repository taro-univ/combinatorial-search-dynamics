# Phase 2 CPU pilot

このpilotは保存、分割、fit、評価、追跡の結合を検証するもので、研究結果として解釈しない。Python 3.11、uv、local file store以外に必要なサービスはない。

`uv sync --all-extras --locked`の後、dummyは`uv run lsd reproduce-pilot experiment=pilot task=dummy_binary solver=none state_model=baseline storage=local tracking=local`で実行する。既存出力は暗黙に置換せず、`--force`を明示する。pathやinstance数を変える場合はHydra overrideで隔離ディレクトリを指定する。各stageの入出力は[architecture](architecture.md)に記す。

rawはPhase 1形式の4表と`observations.zarr`を`data/raw/pilot/`へ置き、`lsd validate-data --data-dir data/raw/pilot`が検証する。実指標は`data/derived/pilot/metrics.parquet`、run IDとconfig hashは`artifacts/pilot/run.json`、reportと集計CSVは`artifacts/pilot/`へ保存する。instance IDの分割は`data/derived/pilot/splits.json`にhashとともに固定し、test IDをfit metadataへ含めない。

MLflowは`file:./mlruns`を既定とし、実行ごとに1 runを作る。tagにはGit commit/dirty状態、config hash、schema version、task名/version、split hash、探索手法名/revision、DVC revisionを記録する。parameterは解決済みHydra設定、metricはheld-outの指標、artifactは解決済みconfig、split、model、予測、report、provenance、validation結果である。

Phase 2 dummy回帰はHydra override付きCLIと自動テストで維持し、独立したDVC graphは持たない。LLM adapter、内部観測、Optuna、nested CV、remote trackingはPhase 4以降に残す。

## Phase 3 knapsack pilot

この文書の上段はPhase 2時点のdummy手順である。Phase 2 pipelineは`uv run lsd reproduce-pilot experiment=pilot task=dummy_binary solver=none state_model=baseline storage=local tracking=local`で再実行する。Phase 3の既定`dvc.yaml`は9 stageで、`generate_instances → solve_references → collect_trajectories`を含む。

ナップサックpilotは`uv run dvc repro`または`uv run lsd reproduce-pilot experiment=knapsack_pilot task=knapsack solver=ortools_cp_sat state_model=objective_gap`で実行する。rawのPhase 1必須4表とZarr、任意拡張の参照表は`data/raw/knapsack_pilot/`、評価は`data/derived/knapsack_pilot/metrics.parquet`、reportは`artifacts/knapsack_pilot/report.md`に置く。`uv run lsd validate-data --data-dir data/raw/knapsack_pilot`でfeasibility、solver結果、checkpoint gapまで検査する。solver単体はinstances生成後に`uv run lsd solve-references experiment=knapsack_pilot task=knapsack solver=ortools_cp_sat`を実行する。

Phase 2/3の出力pathは分離し、Phase 3だけをルートのDVC graphで管理する。solver設定の変更はsolve_references以降だけを再計算し、参照値の証明がないinstanceは真のgap学習・評価に入れない。OR-ToolsはCPUだけで動き、LLM、GPU、remoteは不要である。

成功予測は`uv run lsd analyze-success`で単独再生成できる。入力はraw 4表、参照解、`splits.json`、Hydraのfeatures/evaluation設定で、raw軌跡は変更しない。出力は`success_features.parquet`、`instance_features.parquet`、`success_predictions.parquet`と、モデル・選択履歴・比較表・解決済み分析configである。
