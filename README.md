# LLM Search Dynamics

LLMによる組合せ最適化の探索過程を記録・解析する研究基盤です。現在はPhase 1まで実装済みで、Phase 0の開発骨格に加えて、決定的ID、version付きParquet/Zarr保存、DuckDB view、データ検証を提供します。LLM/API接続、モデル呼び出し、実験データ生成はまだ実装していません。

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
uv run pytest tests/unit tests/integration
uv run lsd doctor
uv run lsd validate-repository
```

`doctor`はPython 3.11、uv、リポジトリ構成、Hydra設定を検査します。API key、LLMサーバー、GPU、DVC remoteは検査しません。`validate-repository`はPhase 1で実在するファイルとHydraのdefault compositionを検証します。

Phase 1データは、4個のParquetファイルとZarr storeがあるディレクトリを指定して検証します。

```bash
uv run lsd validate-data --data-dir /path/to/dataset
```

`--data-dir`を省略するとHydraの`storage.data_dir`を使用します。対象が存在しない、または空の場合は非0で終了します。空を意図的に許容する構成確認だけを行う場合は`--allow-empty`を明示してください。

```bash
uv run lsd validate-data --data-dir /path/to/empty-directory --allow-empty
```

## 設定

`configs/config.yaml`をHydra compositionの入口とし、研究条件を分類別YAMLに置きます。storageはPhase 1で実装済みです。task、LLM、生成、状態モデルなどは引き続き`implemented: false`のplaceholderです。API keyやtokenはHydra YAMLに保存しません。Phase 1の実行に`.env`は不要です。

## Pilotとデータ

Phase 1は保存APIだけを提供し、pilot実行とデータ生成はまだ行いません。問題・生成器・pipelineを実装するPhase 2以降でコマンドと取得手順を追加します。現在の`experiment/pilot.yaml`はHydra compositionを検証するための設定で、研究結果やデータを生成しません。

## Codex CLIについて

Codex CLIはリポジトリ開発を補助する外部ツールであり、研究対象LLMのadapterではありません。必要な開発者だけが別途導入し、`./scripts/run_codex.sh`または`lsd codex -- <prompt>`から利用できます。Phase 0のセットアップとCIには含まれません。

## 構成

- `configs/`: Hydra設定の正本
- `src/llm_search_dynamics/identifiers.py`: canonical JSONによるID生成
- `src/llm_search_dynamics/data/`: schema、Parquet、Zarr、DuckDB、検証
- `tests/unit/`, `tests/integration/`: CPUだけで動く検証
- `docs/adr/`: 技術判断
- `docs/architecture.md`: Phase 0の構成
- `docs/experiment-protocol.md`: 実験計画の雛形
- `docs/data-dictionary.md`: Phase 1のtable・配列・対応規則

全体仕様とPhase 2以降の計画は[リポジトリ仕様書](docs/repository-specification.md)を参照してください。

保存データは将来DVCでversion管理する対象ですが、Phase 1では`dvc.yaml`、pipeline、remoteを作りません。`data/`、Parquet、Zarrは引き続きGit管理対象外です。
