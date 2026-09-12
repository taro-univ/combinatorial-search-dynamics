# Architecture

Phase 0は、Python 3.11のsrc layout、uvによる依存固定、Hydra設定、CLI検証、テスト、CIからなる骨格である。Phase 1はこの上に、決定的IDとversion付き保存・検証境界を追加する。研究処理はまだ実装しない。

`configs/config.yaml`が設定合成の入口であり、experiment、task、llm、generation、observation、state model、dynamics、evaluation、storageを分類する。storageはPhase 1で実装済みとなり、repository-relativeなraw、interim、derived、artifact、Parquet、Zarr pathを定義する。他のplaceholderの`implemented: false`は、その構成が予約済みで実行機能を持たないことを示す。秘密情報は設定に含めない。

`src/llm_search_dynamics/config.py`がHydra設定の合成、解決、および構成検証を担う。`src/llm_search_dynamics/cli.py`はこの機能を`lsd doctor`と`lsd validate-repository`として公開し、Phase 1データ検証を`lsd validate-data`として公開する。

`identifiers.py`はcanonical JSONとSHA-256によるID生成を一元化する。`data/schemas.py`のPyArrow schemaが表構造とschema versionの正本である。`data/parquet.py`は検証済みtableの非破壊的なatomic writeを担い、`data/zarr.py`はcheckpointに対応する多次元配列とmaskをZarr v3へ保存する。

DuckDBはin-memory connection上のviewだけを作り、Parquetを複製しない。`analysis_checkpoints`は`instance_id`と`trial_id`を使ってinstances、trials、checkpointsをjoinする。metricsは行数を不自然に増やさないよう独立viewとする。

`data/validation.py`はtable単体、外部キー、順序、status、Zarr対応をまとめて検証し、構造化されたissue一覧を返す。データは将来DVC管理対象になるが、DVC pipelineはPhase 2以降で追加する。生成、状態モデル、力学、評価も後続Phaseの独立した境界として追加する。詳細は[リポジトリ仕様書](repository-specification.md)を正本とする。
