# 古典探索基盤への移行仕様

## 目的

LLM固有の生成基盤を、複数の古典探索アルゴリズムを同じ条件で実行・比較できる基盤へ置き換える。既存のtask、参照solver、Parquet/Zarr、split、MLflow、DVCは再利用する。既存rawデータは変更しない。

## 名前と構成の変更

| 現在 | 移行後 |
|---|---|
| `configs/llm/` | `configs/search_method/` |
| `configs/generation/` | `configs/search/` |
| `generation/` | `search/` |
| `MockGenerator` | `SearchAlgorithm`実装 |
| `llm_name` | `search_method_name` |
| `llm_revision` | `search_method_revision` |
| `max_new_tokens` | `budget_limit` |
| `generated_token_index` | `budget_used` |
| token/step budget | candidate-evaluation budget |

Python package名とリポジトリ名は、この移行では変更しない。新規書込みはschema version 2とし、version 1は読取り互換だけを残す。

## 探索インターフェース

`SearchAlgorithm`は、task、instance、初期状態、乱数seed、共通の予算管理器を受け取り、trialとcheckpointを返す。最初の問題は単一制約0-1ナップサックとし、次の3手法だけを実装する。

- randomized first-improvement hill climbing
- simulated annealing
- short-term tabu search

共通の候補操作は1ビット反転（itemの追加または削除）とする。容量超過候補は評価予算を1消費して棄却する。各instance・trial seedから作った同じランダム実行可能初期状態を3手法へ渡し、paired比較できるようにする。

Hill Climbingはランダム順に候補を調べて最初の改善を採用し、改善候補がなければ終了する。Simulated Annealingは1候補を抽選し、改善は採用、悪化は温度に応じて確率的に採用する。Tabu Searchは直近に反転したitemを固定tenureだけ禁止し、aspirationはglobal best更新時だけ認める。restart、交換近傍、長期記憶は初回に入れない。

候補の実行可能性または目的値を調べた時点で1評価とする。全近傍を調べれば近傍数だけ消費する。初期状態と参照solverの計算は探索予算に含めない。分析用に後計算する特徴量も含めないが、将来オンライン制御に使う場合は計算に必要な候補評価を予算へ含める。

予算の加算と上限判定は共通の`BudgetedEvaluator`に集約し、探索アルゴリズムが直接目的値を評価しない構造にする。

## 保存項目

trialには最低限、次を保存する。

- `search_method_name`, `search_method_revision`
- `search_seed`
- `budget_type=candidate_evaluations`, `budget_limit`
- 解決済み探索パラメータ
- success、終了理由、実行時間

checkpointには最低限、次を保存する。

- `budget_used`, `remaining_budget`
- `decision_step`, `accepted_moves`, `rejected_moves`
- 現在状態、目的値、参照最適値に対するgap
- terminal判定と直前の操作

`checkpoint_index`は記録順であり、予算として解釈しない。

## Config

実行条件の正本はHydra configとし、Pythonへ値を直書きしない。

```text
configs/
├── experiment/
├── task/
├── solver/
├── search_method/
│   ├── randomized_first_improvement.yaml
│   ├── simulated_annealing.yaml
│   └── tabu.yaml
├── search/
│   └── default.yaml
├── features/
└── evaluation/
```

`search/default.yaml`は`budget_type`、問題サイズから予算を作る倍率、trial数、初期状態生成、checkpoint方針を持つ。アルゴリズム固有値（温度、冷却率、tabu tenure等）は`search_method`側に置く。解決済みconfigを各runのartifactとして保存する。

参照解法は既存のOR-Toolsだけを使う。動的計画法、LP緩和、価値重量比Greedy、Greedy rounding、Newton系、ILSは初回実験の対象外とする。

## 実装順序

1. schema version 2とversion 1読取り互換を追加する。
2. `BudgetedEvaluator`と`SearchAlgorithm` protocolを追加する。
3. 現在のmock探索を共通インターフェースへ移し、回帰結果を確認する。
4. 古典探索アルゴリズムとconfigを追加する。
5. pipeline、CLI、DVC、検証、report、文書、テストを新名称へ移す。
6. LLM固有configと実行経路を削除する。

## 完了条件

- 同一instance・seed・configから同一軌跡を再現できる。
- 全アルゴリズムで予算超過が起きない。
- 候補評価数、移動数、実時間を別々に検証できる。
- rawから成功予測用checkpointを再生成できる。
- `pytest`、データ検証、DVCの再実行が成功する。
