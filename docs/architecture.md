# Architecture

Phase 0のPython 3.11/src layout、uv、Hydra、CLI、CI、Phase 1の保存・検証API、Phase 2のCPU dummy pilotを維持する。Phase 3は0-1ナップサックと独立したOR-Tools CP-SAT参照solverを追加する。どちらのpilotも研究結論を出す実験ではない。

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

## Phase 3: knapsack reference pipeline

`configs/config.yaml`の既定はknapsack pilotで、solver groupを追加した。ルートの`dvc.yaml`はPhase 3の9 stageを持つ。Phase 2 dummy回帰はHydra override付きCLIと自動テストで維持し、追加のルート階層やDVC graphは作らない。solve_referencesはinstancesとtask/solver設定から`reference_solutions.parquet`を作る。collect以降はこの参照表に明示的に依存する。solver設定だけの変更はsolve_references以降、instance生成条件の変更はgenerate_instances以降、report体裁だけの変更はbuild_reportを再実行する。DVCの固定pathはHydra storageの既定pathを投影したもので、CLIのpath overrideはDVC graphへ反映されない。

`data/raw/knapsack_pilot/`はPhase 1の4表、Zarrと任意拡張表`reference_solutions.parquet`を同居させた単一snapshotである。knapsackと判定した場合だけ参照表を必須とし、dummyの4表/Zarr snapshotはそのまま受理する。raw metricsは空schema、実評価値はderivedに置く。Zarrは選択中item vectorの`external_state`を保存し、LLM内部状態を偽造しない。参照とtrialは`instance_id`、trialとcheckpointは`trial_id`で結合する。

Taskは目的の向き、改善、参照値に対するsuccess、状態復元を定義する。solverはTaskと独立してstatus、best feasible value、best bound、証明済みoptimal value、gap、solutionをimmutable結果として返す。mock generatorは参照解のaction列を受け取らず、合法actionと現在の目的値だけで探索する。参照はsuccess/gap/評価にのみ使用し、solver解はTaskでfeasibilityと目的値を再計算する。[ADR-008](adr/ADR-008-ortools-solver-abstraction.md)に数値の定義を記す。

ナップサックの特徴は現在の外部stateと証明済み参照に対するgapである。`objective_gap` state modelは事前固定binを使い、fitはtrainの証明済みinstanceだけに制限する。未証明instanceを0へ変換しない。testはfit metadataと交差しないことを評価前に確認する。問題固有のtest指標はtrial、instance、checkpointの単位を分け、timeoutや未証明をsolver率の母数に残す。真のgapだけは証明済み参照に限定する。

仕様書の完全pipelineの`tune_model`、LLM adapter、prompt versioning、内部観測、Optuna、nested CV、本評価はPhase 4以降に残す。詳細は[リポジトリ仕様書](repository-specification.md)を正本とする。
