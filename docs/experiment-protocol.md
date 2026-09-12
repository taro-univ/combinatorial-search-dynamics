# Experiment Protocol

この文書を実験ごとに複製し、実行前に確定する。Phase 2のdummy binary-target pilotはpipeline検証専用であり、研究結論を出す実験ではない。実問題での研究質問・仮説は以下の欄を事前に確定する。

## Research question

検証する研究質問を記入する。

## Hypotheses

主要仮説、副次仮説、反証条件を記入する。

## Evaluation unit

Phase 2での独立評価単位は`instance_id`である。予測指標はtransitionで算出し、予測JSONのinstance IDからinstance単位へ再集計できる。実研究での評価単位と相関の扱いは事前記入する。

## Data split

Phase 2ではinstance IDをseed 300で決定的に並べ替え、比率0.6/0.2/0.2でtrain/validation/testへ分ける（各split最低1 instance）。splitはhash付きJSONとして固定する。testとvalidationはfitに渡さず、Phase 2でモデル選択はしない。実研究の層別化と複数分割の方針は事前記入する。

## Primary metrics

Phase 2のbaseline stateは現在のHamming距離（0がsuccess）、dynamicsは平滑化0.5の一次Markovモデル。testでone-step NLL（確率を下限`1e-12`にclipした負の自然対数）、next-state accuracy、次状態がsuccessの事象のBrier scoreをtransition平均で算出し、transition/instance/trial数を併記する。これらはpilotの動作検証であり研究上の主要指標ではない。実研究の主要・副次指標、不確実性、比較対象は事前記入する。

## Final evaluation rule

testデータは構成の選択、前処理のfit、仮説の修正に使わず、構成固定後の最終評価まで使用しない。Phase 2はtrainだけで状態と遷移をfitし、validationは保持するだけである。実研究でのtest評価の回数と例外時の扱いは事前記入する。

## Seeds and reproduction

`configs/experiment/pilot.yaml`がinstance生成100、mock sampling 200、分割300、表現400、力学500の用途別seedの正本である。表現・力学は決定的でseedを記録するだけで乱数は使わない。Python 3.11で`uv sync --all-extras --locked`し、隔離pathで`lsd reproduce-pilot`と`lsd validate-data --data-dir <raw>`を実行する。既定pathなら`dvc repro`を使い、2回目のskipを確認する。詳細は[pilot手順](pilot.md)。
