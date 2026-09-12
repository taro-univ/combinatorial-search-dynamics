# Phase 2 CPU pilot

このpilotは保存、分割、fit、評価、追跡の結合を検証するもので、研究結果として解釈しない。Python 3.11、uv、local file store以外に必要なサービスはない。

`uv sync --all-extras --locked`の後、`uv run dvc repro`で8 stageを実行し、`uv run dvc status`と2回目の`uv run dvc repro`で変更のないstageがskipされることを確認する。`uv run lsd reproduce-pilot`は同じstage APIを一度に実行するCLIである。既存出力は暗黙に置換せず、`--force`を明示する。Hydra overrideでpathやinstance数を変える場合、DVC stageの既定pathとは別の隔離ディレクトリを指定する。各stageの入出力は[architecture](architecture.md)に記す。

rawはPhase 1形式の4表と`observations.zarr`を`data/raw/pilot/`へ置き、`lsd validate-data --data-dir data/raw/pilot`が検証する。実指標は`data/derived/pilot/metrics.parquet`、run IDとconfig hashは`artifacts/pilot/run.json`、reportと集計CSVは`artifacts/pilot/`へ保存する。instance IDの分割は`data/derived/pilot/splits.json`にhashとともに固定し、test IDをfit metadataへ含めない。

MLflowは`file:./mlruns`を既定とし、実行ごとに1 runを作る。tagにはGit commit/dirty状態、config hash、schema version、task名/version、split hash、mock generator名/revision、DVC revisionが確定できない理由を記録する。parameterは解決済みHydra設定、metricはheld-outの3指標とtransition数、artifactは解決済みconfig、split、model 2種、metrics Parquet、予測/集計、report、provenance、validation結果である。実値のファイルを別途保存するためMLflowのみを正本にしない。

`dvc.yaml`のpathはHydra storage既定値に合わせた静的graphで、report体裁の変更はbuild_report、モデル条件の変更はfit_model以降、featureコードの変更はextract_features以降を再計算する。実問題solver、LLM adapter、内部観測、Optuna、nested CV、remote trackingはPhase 3以降に残す。
