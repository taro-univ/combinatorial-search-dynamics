# Architecture

Phase 0のPython 3.11/src layout、uv、Hydra、CLI、CIとPhase 1の決定的ID、PyArrow schema、Parquet/Zarr I/O、DuckDB view、validationを維持する。Phase 2はCPU専用のdummy pilotだけを追加する。研究結論を出す実験ではない。

`configs/config.yaml`が設定合成の入口であり、experiment、task、llm（mock識別）、generation、observation、state_model、dynamics、evaluation、storage、trackingを分類する。Phase 2で実装したgroupのみ`implemented: true`とする。storageはrepository-relativeなraw、interim、derived、artifact、Parquet、Zarr pathを定義し、trackingはlocal file URIを定義する。秘密情報は設定・provenanceに含めない。

`src/llm_search_dynamics/config.py`はHydra設定の合成・解決・構成検証を担う。CLIは`lsd doctor`、`lsd validate-repository`、Phase 1データ検証の`lsd validate-data`とPhase 2のstage commandを公開する。末尾のHydra overrideを各stageへ渡す。

`identifiers.py`はcanonical JSONとSHA-256によるID生成を一元化する。`data/schemas.py`のPyArrow schemaがversion `1`と表構造の正本である。`data/parquet.py`は非破壊的なatomic writeを担い、`data/zarr.py`はcheckpointに対応する値とmaskをZarr v3へ保存する。Phase 2は既存内部観測groupを後方互換に保ち、`external_state` groupのbit vectorだけを保存する。内部観測を偽造しない。

DuckDBはin-memory connection上のviewだけを作り、Parquetを複製しない。`analysis_checkpoints`は`instance_id`と`trial_id`を使ってinstances、trials、checkpointsをjoinする。metricsは行数を不自然に増やさないよう独立viewとする。

`data/validation.py`はtable単体、外部キー、順序、status、Zarr対応をまとめて検証し、構造化issueを返す。rawは`data/raw/pilot/`に4表とZarrを同居させる単一の検証可能なsnapshotを採用する。rawの`metrics.parquet`は空schema付きで、実際のtest評価値は`data/derived/pilot/metrics.parquet`へ置く。異なる役割の2表を明示的に区別し、他のデータを無秩序に複製しない。collectはmetricsを最後に確定し、不完全なsnapshotを成功扱いしない。

`tasks/dummy_binary.py`はbinary-target問題、`generation/mock.py`はseed付き探索、`features/external.py`は現在の外部状態のみの特徴、`data/splits.py`はinstance ID単位の決定的分割を担当する。train以外をfit APIへ渡すと拒否する。`state_models/baseline.py`は現在のHamming距離を離散状態にし、`dynamics/transition.py`はtrainの一次Markov countと平滑化を使い成功状態0を吸収とする。モデルにfit split/hash/instance IDを記録し、evaluateはtestとの交差を拒否する。

`pipeline/stages.py`は次の独立した再計算境界を持つ。各stageは必要な前段artifactと出力の衝突を検査し、`reproduce-pilot`は内部APIを順番に呼ぶ。

| DVC stage / CLI | 入力 → 出力 |
|---|---|
| generate_instances / generate-instances | Hydra seed・dummy task → raw instances |
| collect_trajectories / collect | raw instances・mock条件 → raw trials、checkpoints、外部Zarr、空metrics |
| prepare_trajectories / prepare | 検証済みraw → interim trajectories |
| extract_features / extract | interim trajectories → derived features |
| make_splits / split | instance ID・seed・比率 → derived split/hash |
| fit_model / fit | train features → model artifacts、derived state assignments |
| evaluate_model / evaluate | 固定model・held-out test → derived metrics/predictions、MLflow、provenance |
| build_report / report | raw統計・評価summary → Markdown/CSV、MLflow artifacts |

MLflowはlocal file storeにrun・parameter・metric・tag・artifactを記録し、derived Parquet/JSON/CSVから再集計できる。DVCは生成データcacheとstage依存関係を管理し、Gitはコード・設定・DVC metadataを管理する。`dvc.yaml`内のpathはHydra storage既定値への投影であり、既定path変更時は両方更新する。個別CLIのpath overrideはDVC graphを変更しない。詳細は[ADR-006](adr/ADR-006-git-dvc-mlflow-responsibilities.md)を参照する。

仕様書の完全pipelineのsolve_referencesとtune_model、実問題、LLM adapter、内部観測、Optuna、nested CV、本評価はPhase 3以降に残す。詳細は[リポジトリ仕様書](repository-specification.md)を正本とする。
