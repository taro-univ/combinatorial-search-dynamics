# B2・M2・M3 成功予測実験仕様

## 研究質問

0-1ナップサックに対する3種類の探索で、残り候補評価予算内に大域最適値へ到達する確率は、budgetとgapだけでどこまで予測でき、局所地形または探索履歴を一つ追加するとどれだけ改善するか。さらに、有効な量を少数組み合わせたモデルはB2を上回るか。

対象はRandomized First-Improvement Hill Climbing、Simulated Annealing、Short-term Tabu Searchである。参照最適値はOR-Toolsだけで求める。それ以外の厳密解法、緩和法、非軌跡ベースラインは初回に含めない。

予測器は比較を明確にするため、全条件で同じ正則化ロジスティック回帰を使う。

## 目的変数とベースライン

評価例は非終端checkpoint、目的変数はその後`budget_limit`以内に証明済み最適値へ到達したかである。

**B2**

- `remaining_budget_fraction = remaining_budget / budget_limit`
- `relative_objective_gap`

## 単体追加候補

各モデルは、まずB2へ下記の1変数だけを追加する。

**M2：現在状態と局所地形**

- `M2_distance`：最近傍の最適解までの正規化Hamming距離（分析用oracle）
- `M2_improving_fraction`：改善する合法近傍の割合
- `M2_best_gain`：近傍内の最大正規化改善量
- `M2_local_entropy`：改善・同値・悪化・実行不可能の近傍割合のentropy

**M3：checkpoint以前の探索履歴**

- `M3_progress_rate`：直近windowの正規化改善速度
- `M3_stagnation`：最後のbest更新からの消費予算率
- `M3_revisit`：既訪問状態への再訪率
- `M3_best_update_rate`：過去のbest更新回数を消費予算で割った値
- `M3_visit_entropy`：過去の訪問状態分布のentropy

未来のcheckpointは特徴計算に使わない。M2の近傍は全itemの1ビット反転とし、容量超過も`M2_local_entropy`の実行不可能カテゴリへ含める。正のitem valueでは同値な1ビット反転が発生しないため、neutral fractionは初回候補から除外する。M3のwindowと正規化方法はconfigで固定する。

## 選択手順

1. B2をfitする。
2. `B2 + M2候補1個`と`B2 + M3候補1個`を全候補で評価し、単体の増分を記録する。
3. validation Brier scoreによる前進選択でM2を組み合わせる。追加改善がconfigの`min_delta`未満、または`max_features`到達で停止する。
4. 選ばれたM2モデルへM3を一つずつ追加し、同じ規則でM3の組合せを選ぶ。
5. 構成固定後、testを一度だけ評価し、B2との差と信頼区間を報告する。

単体結果は最終組合せに採用されなくても保存する。testを特徴選択に使わない。

## 分割と評価

- split単位：`instance_id`
- 同一instanceの全trial/checkpointを同じsplitへ置く
- 主要比較は探索アルゴリズム別に行う
- 同じinstance・trialでは3手法へ同じ初期状態を与える
- fit時は各trialのcheckpoint weight合計を1にする
- 主指標：instance-macro Brier score
- 副指標：NLL、AUROC、calibration、trial/instance成功率
- 不確実性：instance単位bootstrapによるpaired信頼区間

精度向上は追加の予測情報を示すもので、因果効果とは解釈しない。

## 問題選定用の記録

`instance_features.parquet`へ、状態特徴とは分けて次を保存する。

- 問題サイズ、容量比、weight/valueの平均・変動係数・相関
- value/weight比の要約、支配されるitemの割合
- samplingによる目的値分布、FDC、自己相関、局所最適率
- 改善近傍率、局所entropyのinstance内平均と分散
- sampling seed、サンプル数、近傍定義、計算version

初回の成功予測には原則入れず、難易度の層別化と将来の問題横断モデルに利用する。

## Configと成果物

`configs/experiment/`にinstance数、trial数、seed、splitを置き、`configs/search/`に主予算、`configs/search_method/`に探索パラメータ、`configs/features/`に候補特徴、window、sampling条件、`configs/evaluation/`に選択規則と指標を置く。

raw軌跡は上書きしない。特徴量、各単体モデル、選択履歴、最終モデル、予測、比較表、解決済みconfigをderived/artifactsへanalysis version付きで保存する。

## 実験開始条件

- 古典探索基盤への移行仕様が完了している。
- 予算計数と全特徴量の単体テストがある。
- pilotで成功と失敗の両方が発生する。
- split、候補特徴、選択規則、test評価手順が実行前に固定されている。

## 実装上の固定定義

- `M2_distance`はOR-Toolsで目的値を参照最適値に固定し、現在状態とのHamming距離を最小化して求める。
- `M2_best_gain`は`max(0, 近傍価値−現在価値) / max(|最適値|, epsilon)`。
- `M2_local_entropy`は全itemを分母に、改善・同値・悪化・容量超過の4分類から自然対数で求める。
- `M3_progress_rate`はwindow内の目的値差を最適値で割り、さらに消費予算率で割る。悪化時は負値を許す。
- `M3_visit_entropy`は履歴長の対数で正規化する。初期状態しかない場合は0とする。
- fit標準化量はtrainだけで求め、validationは選択、testは固定後の評価だけに使う。

## 再現手順

```bash
uv run lsd reproduce-pilot
uv run lsd validate-data --data-dir data/raw/knapsack_pilot
uv run dvc repro
```

特徴分析だけの再生成は`uv run lsd analyze-success --force`を使う。主要結果は`artifacts/knapsack_pilot/success_feature_comparison.md`、全選択履歴は`success_feature_selection.json`、問題選定用の量は`data/derived/knapsack_pilot/instance_features.parquet`で確認する。
