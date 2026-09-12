# LLM Search Dynamics

組合せ最適化の探索を記録・解析する研究基盤です。Phase 2では研究結果を出すためではないCPU専用のdummy pilotを実行できます。seedからbinary-target instanceを生成し、mock探索をPhase 1形式で保存して、instance単位の分割、Hamming距離の状態表現、一次Markovモデル、held-out評価、MLflow記録、Markdown reportまで完走します。外部API、LLM、GPU、DVC remote、MLflow serverは不要です。

## セットアップ

Pythonは3.11を使用します。uvを導入済みなら、lock済みの全依存を次で同期できます。

```bash
uv sync --all-extras --locked
```

uvの導入も含める場合は、次のスクリプトを使います。このスクリプトは`.env`を作成・変更せず、Codex CLIも導入しません。

```bash
./scripts/bootstrap.sh
```

依存を変更した場合は`uv lock`で`uv.lock`を更新し、コードと一緒に管理します。

## ローカル検証

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit tests/integration tests/regression
uv run lsd doctor
uv run lsd validate-repository
uv run dvc dag
```

`doctor`はPython 3.11、uv、リポジトリ構成、Hydra設定を検査します。API key、LLMサーバー、GPU、DVC remoteは検査しません。`validate-repository`は実在するファイルとHydraのdefault compositionを検証します。

Phase 1形式のデータは、4個のParquetファイルとZarr storeがあるディレクトリを指定して検証します。

```bash
uv run lsd validate-data --data-dir /path/to/dataset
```

`--data-dir`を省略するとHydraの`storage.data_dir`を使用します。対象が存在しない、または空の場合は非0で終了します。空を意図的に許容する構成確認だけを行う場合は`--allow-empty`を明示してください。

```bash
uv run lsd validate-data --data-dir /path/to/empty-directory --allow-empty
```

## 設定

`configs/config.yaml`をHydra compositionの入口とし、研究条件を分類別YAMLに置きます。Phase 2で実装したdummy task、mock識別、生成、外部観測、状態モデル、力学、評価、storage、local trackingだけを`implemented: true`にしています。API keyやtokenはHydra YAMLに保存しません。pilotに`.env`は不要です。

## Pilotとデータ

```bash
uv run lsd reproduce-pilot
uv run lsd validate-data --data-dir data/raw/pilot
uv run dvc repro
uv run dvc status
uv run dvc repro  # 未変更なら全stageをskip
```

既存出力との衝突時は非0終了します。再生成を明示する場合は`--force`を指定します。DVC stageは自身の出力を`--force`で再計算し、必要なstageだけ実行します。DVCの既定出力を先にCLIで作った場合も、`dvc repro`が所有する出力は明示的に再生成します。

個別stageも順に単独実行できます。前段artifactが欠ける場合は失敗します。

```bash
uv run lsd generate-instances
uv run lsd collect
uv run lsd prepare
uv run lsd extract
uv run lsd split
uv run lsd fit
uv run lsd evaluate
uv run lsd report
```

CLI末尾のHydra overrideで小規模な別pilotを隔離できます。

```bash
uv run lsd reproduce-pilot experiment.instance_count=12 task.bit_count=6 generation.trials=2 \
  storage.raw_data_dir=/tmp/lsd-pilot/raw storage.interim_data_dir=/tmp/lsd-pilot/interim \
  storage.derived_data_dir=/tmp/lsd-pilot/derived storage.artifact_dir=/tmp/lsd-pilot/artifacts \
  tracking.tracking_uri=file:/tmp/lsd-pilot/mlruns
```

local MLflow runは任意で`MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri file:./mlruns`から確認できます。pilot自体にserver起動は不要です。MLflow 3.16以降で必要なfile-store opt-inはtracking adapterがpilotのプロセス内で設定します。

`llm/mock.yaml`はPhase 1の`trials.llm_name`との互換用識別でありLLM adapterではありません。`temperature`は探索確率`1 - greedy_probability`、`max_new_tokens`はstep budgetに対応します。Phase 3以降のsolver、実問題、LLM adapter、prompt versioning、内部観測、Optuna、nested CV、本評価は未実装です。

## Codex CLIについて

Codex CLIはリポジトリ開発を補助する外部ツールであり、研究対象LLMのadapterではありません。必要な開発者だけが別途導入し、`./scripts/run_codex.sh`または`lsd codex -- <prompt>`から利用できます。Phase 0のセットアップとCIには含まれません。

## 構成

- `configs/`: Hydra設定の正本
- `src/llm_search_dynamics/identifiers.py`: canonical JSONによるID生成
- `src/llm_search_dynamics/data/`: schema、Parquet、Zarr、DuckDB、検証
- `tests/unit/`, `tests/integration/`, `tests/regression/`: CPUだけで動く検証
- `docs/adr/`: 技術判断
- `docs/architecture.md`: Phase 0–2の構成
- `docs/experiment-protocol.md`: pilot条件と将来の実験計画
- `docs/data-dictionary.md`: table・配列・対応規則
- `docs/pilot.md`: 再現手順とstage成果物

全体仕様とPhase 3以降の計画は[リポジトリ仕様書](docs/repository-specification.md)を参照してください。

Gitはコード・Hydra設定・文書・`dvc.yaml`・`dvc.lock`を管理します。DVCは生成データcacheと依存関係、MLflowはlocal runを管理し、Hydraが科学条件の正本です。`data/`、`artifacts/`、`mlruns/`、`.dvc/cache/`はGit管理対象外であり、remoteやcredentialは設定しません。rawのPhase 1 snapshotは`data/raw/pilot/`、interimは`data/interim/pilot/`、split・予測・評価値は`data/derived/pilot/`、model・provenance・reportは`artifacts/pilot/`に置きます。
