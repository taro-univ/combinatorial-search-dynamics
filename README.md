# LLM Search Dynamics

LLMによる組合せ最適化の探索過程を記録・解析する研究基盤です。現在はPhase 0の骨格で、Pythonパッケージ、再現可能な依存、Hydra設定、ローカル検証、テスト、CIを提供します。LLM/API接続、モデル呼び出し、実験データ生成はまだ実装していません。

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

## Phase 0の検証

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit tests/integration
uv run lsd doctor
uv run lsd validate-repository
```

`doctor`はPython 3.11、uv、リポジトリ構成、Hydra設定を検査します。API key、LLMサーバー、GPU、DVC remoteは検査しません。`validate-repository`はPhase 0で実在するファイルとHydraのdefault compositionを検証します。

仕様上の将来コマンド名`validate-data`は後方互換のaliasとして残しています。Phase 0では実験データが存在しないため、同じリポジトリ検証だけを行い、その旨を表示します。実データのschema検証はPhase 1で実装します。

## 設定

`configs/config.yaml`をHydra compositionの入口とし、研究条件を分類別YAMLに置きます。現在のtask、LLM、生成、状態モデルなどは`implemented: false`のplaceholderです。API keyやtokenはHydra YAMLに保存しません。Phase 0の実行に`.env`は不要です。

## Pilotとデータ

pilot実行とデータ取得はPhase 0では利用できません。問題・生成器・保存pipelineを実装する後続Phaseでコマンドと取得手順を追加します。現在の`experiment/pilot.yaml`はHydra compositionを検証するための設定で、研究結果やデータを生成しません。

## Codex CLIについて

Codex CLIはリポジトリ開発を補助する外部ツールであり、研究対象LLMのadapterではありません。必要な開発者だけが別途導入し、`./scripts/run_codex.sh`または`lsd codex -- <prompt>`から利用できます。Phase 0のセットアップとCIには含まれません。

## 構成

- `configs/`: Hydra設定の正本
- `src/llm_search_dynamics/`: PythonパッケージとCLI
- `tests/unit/`, `tests/integration/`: CPUだけで動く検証
- `docs/adr/`: 技術判断
- `docs/architecture.md`: Phase 0の構成
- `docs/experiment-protocol.md`: 実験計画の雛形

全体仕様とPhase 1以降の計画は[リポジトリ仕様書](docs/repository-specification.md)を参照してください。
