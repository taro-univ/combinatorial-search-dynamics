# Experiment Protocol

この文書を実験ごとに複製し、実行前に確定する。研究結果や都合のよい条件を事後に書き込むための文書ではない。

## Research question

検証する研究質問を記入する。

## Hypotheses

主要仮説、副次仮説、反証条件を記入する。

## Evaluation unit

独立な評価単位を記入する。既定候補は`instance_id`であり、同一instanceまたはtrialを複数splitへ分散させない。

## Data split

train、validation、testの生成規則、seed、層別化、固定方法を記入する。前処理、状態モデル、ハイパーパラメータ選択はtrain側だけで学習し、validationで選択する。

## Primary metrics

主要指標、集計方法、不確実性の表現、比較対象を実験前に記入する。副次指標と探索的解析は区別する。

## Final evaluation rule

testデータは構成の選択、前処理のfit、仮説の修正に使わず、構成固定後の最終評価まで使用しない。test評価の回数と例外時の扱いを記入する。
