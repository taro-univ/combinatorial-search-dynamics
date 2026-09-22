# Architecture

Phase 0のPython 3.11/src layout、uv、Hydra、CLI、CI、Phase 1の保存・検証API、Phase 2のCPU dummy pilotを維持する。Phase 3は0-1ナップサックと独立したOR-Tools CP-SAT参照solverを追加する。どちらのpilotも研究結論を出す実験ではない。

`configs/config.yaml`が設定合成の入口であり、experiment、task、solver、search_method、search、observation、state_model、dynamics、evaluation、storage、trackingを分類する。storageはrepository-relativeなraw、interim、derived、artifact、Parquet、Zarr pathを定義し、trackingはlocal file URIを定義する。

`src/combinatorial_search_dynamics/config.py`はHydra設定の合成・解決・構成検証を担う。CLIは`csd doctor`、`csd validate-repository`、Phase 1データ検証の`csd validate-data`とPhase 2のstage commandを公開する。末尾のHydra overrideを各stageへ渡す。

`identifiers.py`はcanonical JSONとSHA-256によるID生成を一元化する。`data/schemas.py`のPyArrow schema version 2が新規書込みの正本で、version 1は読取り互換のみである。`data/zarr.py`も新規には`budget_used` indexを書き、version 1の`generated_token_index`を読み込める。

DuckDBはin-memory connection上のviewだけを作り、Parquetを複製しない。`analysis_checkpoints`は`instance_id`と`trial_id`を使ってinstances、trials、checkpointsをjoinする。metricsは行数を不自然に増やさないよう独立viewとする。

`data/validation.py`はtable単体、外部キー、順序、status、Zarr対応をまとめて検証し、構造化issueを返す。rawは`data/raw/pilot/`に4表とZarrを同居させる単一の検証可能なsnapshotを採用する。rawの`metrics.parquet`は空schema付きで、実際のtest評価値は`data/derived/pilot/metrics.parquet`へ置く。異なる役割の2表を明示的に区別し、他のデータを無秩序に複製しない。collectはmetricsを最後に確定し、不完全なsnapshotを成功扱いしない。

`search/`は共通の候補評価予算、初期状態生成、Randomized First-Improvement、Simulated Annealing、Short-term Tabu Searchを実装する。容量超過候補も1評価として数え、全手法が同じ予算管理器を使う。`features/external.py`以降は収集済みの外部状態から再計算する。

`pipeline/stages.py`は次の独立した再計算境界を持つ。各stageは必要な前段artifactと出力の衝突を検査し、`reproduce-pilot`は内部APIを順番に呼ぶ。

| DVC stage / CLI | 入力 → 出力 |
|---|---|
| generate_instances / generate-instances | Hydra seed・dummy task → raw instances |
| collect_trajectories / collect | raw instances・探索条件 → raw trials、checkpoints、外部Zarr、空metrics |
| prepare_trajectories / prepare | 検証済みraw → interim trajectories |
| extract_features / extract | interim trajectories → derived features |
| make_splits / split | instance ID・seed・比率 → derived split/hash |
| analyze_success / analyze-success | raw軌跡・参照解・split → B2/M2/M3特徴、問題特徴、選択履歴、test比較 |
| fit_model / fit | train features → model artifacts、derived state assignments |
| evaluate_model / evaluate | 固定model・held-out test → derived metrics/predictions、MLflow、provenance |
| build_report / report | raw統計・評価summary → Markdown/CSV、MLflow artifacts |

MLflowはlocal file storeにrun・parameter・metric・tag・artifactを記録し、derived Parquet/JSON/CSVから再集計できる。DVCは生成データcacheとstage依存関係を管理し、Gitはコード・設定・DVC metadataを管理する。`dvc.yaml`内のpathはHydra storage既定値への投影であり、既定path変更時は両方更新する。個別CLIのpath overrideはDVC graphを変更しない。詳細は[ADR-006](adr/ADR-006-git-dvc-mlflow-responsibilities.md)を参照する。

## Phase 3: knapsack reference pipeline

`configs/config.yaml`の既定はknapsack pilotである。ルートの`dvc.yaml`は成功分析を含むPhase 3の10 stageを持つ。solve_referencesはinstancesとtask/solver設定から`reference_solutions.parquet`を作り、collect以降はこの参照表へ明示的に依存する。

`data/raw/knapsack_pilot/`はPhase 1の4表、Zarrと任意拡張表`reference_solutions.parquet`を同居させた単一snapshotである。knapsackと判定した場合だけ参照表を必須とし、dummyの4表/Zarr snapshotはそのまま受理する。raw metricsは空schema、実評価値はderivedに置く。Zarrは選択中item vectorの`external_state`を保存し、LLM内部状態を偽造しない。参照とtrialは`instance_id`、trialとcheckpointは`trial_id`で結合する。

Taskは目的の向き、改善、参照値に対するsuccess、状態復元を定義する。solverはTaskと独立して参照解を返す。探索手法はsolverの解ベクトルを受け取らず、参照値はsuccessとgap判定だけに使う。[ADR-008](adr/ADR-008-ortools-solver-abstraction.md)に数値の定義を記す。

ナップサックの特徴は現在の外部stateと証明済み参照に対するgapである。`objective_gap` state modelは事前固定binを使い、fitはtrainの証明済みinstanceだけに制限する。未証明instanceを0へ変換しない。testはfit metadataと交差しないことを評価前に確認する。問題固有のtest指標はtrial、instance、checkpointの単位を分け、timeoutや未証明をsolver率の母数に残す。真のgapだけは証明済み参照に限定する。

checkpoint成功予測は、終端checkpointを除き、train内ではtrialごとのcheckpoint weight合計を等しくする。評価値はinstance内で集約してからinstance間平均を取り、同一instanceの全trialを同じsplitに保つ。今後の基盤変更は[古典探索移行仕様](experiments/classical-search-migration.md)、比較実験は[B2・M2・M3実験仕様](experiments/b2-m2-m3-feature-selection.md)に記す。

既定の`classical_comparison`は同一instance・trial seed・初期状態を3手法へ渡す。`analyze-success`は探索手法別に同じ正則化ロジスティック回帰をfitし、単体追加とvalidation前進選択を行う。testは構成固定後のB2、選択済みM2、最終モデルだけに使う。

仕様書の完全pipelineの`tune_model`、LLM adapter、prompt versioning、内部観測、Optuna、nested CV、本評価はPhase 4以降に残す。詳細は[リポジトリ仕様書](repository-specification.md)を正本とする。
