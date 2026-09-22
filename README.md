# LLM Search Dynamics

組合せ最適化の探索を記録・解析する研究基盤です。0-1ナップサックをOR-Tools CP-SATで参照計算し、古典探索の軌跡を収集する小規模CPU pilotを実行できます。外部API、LLM、GPU、DVC remote、MLflow serverは不要です。

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

Phase 1形式のデータは、4個のParquetファイルとZarr storeがあるディレクトリを指定して検証します。ナップサックdatasetでは同じraw directoryの`reference_solutions.parquet`も必須としてfeasibility、参照値、checkpoint gapを検査します。dummy datasetでは参照表を要求しません。

```bash
uv run lsd validate-data --data-dir /path/to/dataset
```

`--data-dir`を省略するとHydraの`storage.data_dir`を使用します。対象が存在しない、または空の場合は非0で終了します。空を意図的に許容する構成確認だけを行う場合は`--allow-empty`を明示してください。

```bash
uv run lsd validate-data --data-dir /path/to/empty-directory --allow-empty
```

## 設定

`configs/config.yaml`をHydra compositionの入口とし、研究条件を分類別YAMLに置きます。既定はknapsack pilotで、`task=knapsack`、`solver=ortools_cp_sat`、`state_model=objective_gap`です。dummy pilotは明示的なgroup overrideで再実行できます。API keyやtokenはHydra YAMLに保存しません。pilotに`.env`は不要です。

## Pilotとデータ

```bash
uv run lsd reproduce-pilot
uv run lsd validate-data --data-dir data/raw/knapsack_pilot
uv run dvc repro
uv run dvc status
uv run dvc repro  # 未変更なら全stageをskip
```

既存出力との衝突時は非0終了します。再生成を明示する場合は`--force`を指定します。DVC stageは自身の出力を`--force`で再計算し、必要なstageだけ実行します。DVCの既定出力を先にCLIで作った場合も、`dvc repro`が所有する出力は明示的に再生成します。

個別stageも順に単独実行できます。前段artifactが欠ける場合は失敗します。

```bash
uv run lsd generate-instances
uv run lsd solve-references
uv run lsd collect
uv run lsd prepare
uv run lsd extract
uv run lsd split
uv run lsd analyze-success
uv run lsd fit
uv run lsd evaluate
uv run lsd report
```

古典探索基盤の詳細は[移行仕様](docs/experiments/classical-search-migration.md)、次の成功予測比較は[B2・M2・M3実験仕様](docs/experiments/b2-m2-m3-feature-selection.md)に記載しています。

CLI末尾のHydra overrideで小規模な別pilotを隔離できます。

```bash
uv run lsd reproduce-pilot experiment=knapsack_pilot task=knapsack solver=ortools_cp_sat \
  state_model=objective_gap experiment.instance_count=12 task.item_count=6 search.trials=2 \
  storage.raw_data_dir=/tmp/lsd-pilot/raw storage.interim_data_dir=/tmp/lsd-pilot/interim \
  storage.derived_data_dir=/tmp/lsd-pilot/derived storage.artifact_dir=/tmp/lsd-pilot/artifacts \
  tracking.tracking_uri=file:/tmp/lsd-pilot/mlruns
```

local MLflow runは任意で`MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri file:./mlruns`から確認できます。pilot自体にserver起動は不要です。MLflow 3.16以降で必要なfile-store opt-inはtracking adapterがpilotのプロセス内で設定します。

既定は3探索手法のpaired比較です。単独実行は`search_method=randomized_first_improvement`、`simulated_annealing`、`tabu`で切り替えます。主予算は候補解を調べた回数で、`search.budget_limit`がnullなら`search.budget_multiplier × 問題サイズ`です。`analyze-success`はB2・M2・M3の特徴抽出、validation前進選択、固定後のtest評価、問題特徴の記録を行います。

ナップサックの個別solver stageは、instance生成後に実行できます。

```bash
uv run lsd generate-instances experiment=knapsack_pilot task=knapsack
uv run lsd solve-references experiment=knapsack_pilot task=knapsack solver=ortools_cp_sat
uv run lsd reproduce-pilot experiment=knapsack_pilot task=knapsack solver=ortools_cp_sat state_model=objective_gap
```

Phase 2のdummy回帰は別pathを使います。既存出力がある場合は明示的に`--force`を使うか、隔離pathを指定します。

```bash
uv run lsd reproduce-pilot experiment=pilot task=dummy_binary solver=none state_model=baseline storage=local tracking=local
```

既定`dvc.yaml`はPhase 3の9 stage graphです。Phase 2 dummy回帰は上記のHydra override付きCLIと自動テストで維持します。DVCは固定された既定pathを使うため、CLIのpath overrideはDVC graphへ反映されません。生成物はナップサックなら`data/raw/knapsack_pilot/`、`data/interim/knapsack_pilot/`、`data/derived/knapsack_pilot/`、`artifacts/knapsack_pilot/`、MLflowは`mlruns/`に置かれます。LLM接続、prompt versioning、内部状態、Optuna、nested CV、本評価はPhase 4以降です。

## Codex CLIについて

Codex CLIはリポジトリ開発を補助する外部ツールであり、研究対象LLMのadapterではありません。必要な開発者だけが別途導入し、`./scripts/run_codex.sh`または`lsd codex -- <prompt>`から利用できます。Phase 0のセットアップとCIには含まれません。

## 構成

- `configs/`: Hydra設定の正本
- `src/llm_search_dynamics/identifiers.py`: canonical JSONによるID生成
- `src/llm_search_dynamics/data/`: schema、Parquet、Zarr、DuckDB、検証
- `tests/unit/`, `tests/integration/`, `tests/regression/`: CPUだけで動く検証
- `docs/adr/`: 技術判断
- `docs/architecture.md`: Phase 0–3の構成
- `docs/experiment-protocol.md`: pilot条件と将来の実験計画
- `docs/experiments/`: 個別実験の仮説、比較条件、反証条件
- `docs/data-dictionary.md`: table・配列・対応規則
- `docs/pilot.md`: 再現手順とstage成果物

全体仕様とPhase 4以降の計画は[リポジトリ仕様書](docs/repository-specification.md)を参照してください。

Gitはコード・Hydra設定・文書・`dvc.yaml`/`dvc.lock`を管理します。DVCは生成データcacheと依存関係、MLflowはlocal runを管理し、Hydraが科学条件の正本です。`data/`、`artifacts/`、`mlruns/`、`.dvc/cache/`はGit管理対象外であり、remoteやcredentialは設定しません。Phase 2のsnapshotは`data/raw/pilot/`、Phase 3は`data/raw/knapsack_pilot/`に置き、それぞれ同名のinterim/derived/artifact directoryを使用します。
