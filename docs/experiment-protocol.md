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

`configs/experiment/pilot.yaml`がinstance生成100、mock sampling 200、分割300、表現400、力学500の用途別seedの正本である。表現・力学は決定的でseedを記録するだけで乱数は使わない。Python 3.11で`uv sync --all-extras --locked`し、隔離pathで`csd reproduce-pilot`と`csd validate-data --data-dir <raw>`を実行する。既定pathなら`dvc repro`を使い、2回目のskipを確認する。詳細は[pilot手順](pilot.md)。

## Phase 3 knapsack pilot protocol

Phase 3は最初の実問題adapterを通すpipeline検証であり、本研究の最終結論を出す実験ではない。独立評価単位とsplit単位は`instance_id`。同一instanceの全trial/checkpointを同じsplitに置き、testを状態binのfit、遷移確率のfit、設定選択に使用しない。

instance生成条件の正本は`configs/task/knapsack.yaml`で、既定は6 item、weight 1–9、value 1–15、capacityはweight合計の0.5倍を整数化し全itemが入らないようにする。`configs/experiment/knapsack_pilot.yaml`はinstance数9、生成seed 1100、sampling seed 1200、split seed 1300、表現seed 1400、遷移seed 1500を定義する。solver条件は`configs/solver/ortools_cp_sat.yaml`で、CP-SATの最大実行時間5秒、1 worker、seed 3001、gap denominator epsilon 1e-9、探索ログ無効を既定とする。値はHydra overrideで変更でき、解決済みconfigをMLflow artifactに記録する。

参照表で`OPTIMAL`と証明されたinstanceだけが真のoptimal値を持つ。`FEASIBLE`やtimeoutを削除せずstatus/率の母数に残し、真のgap指標とgap状態モデルのfit/evaluateからは除外する。checkpointは必ずfeasibleで、探索手法は参照解のaction列を見ず、参照値をsuccessとgap判定にだけ使う。

状態表現は証明済みoptimal値に対する現在の相対gapを事前固定binへ割り当て、gap 0を吸収状態0とする。一次Markovモデルはtrain trajectoryのみからfitする。held-out testではone-step NLL、next-state accuracy、success-state Brierをtransition平均で算出する。問題固有にはtrial/instance success rate、最終価値、最終absolute/relative gap、best-so-far価値、最適到達step、feasible checkpoint率、solver status件数、timeout率、証明率を出す。gapの分母は証明済みinstanceのみで、単位と計算式は[data dictionary](data-dictionary.md)に記す。solver runtimeとmock探索runtimeは別に扱う。

再現手順はPython 3.11で`uv sync --all-extras --locked`し、隔離pathを指定した`csd reproduce-pilot experiment=knapsack_pilot task=knapsack solver=ortools_cp_sat state_model=objective_gap`、続いて`csd validate-data --data-dir <raw>`を実行する。既定pathのDVC graphなら`dvc repro`を実行し、2回目で不要なstageがskipされることを確認する。dummy回帰は`csd reproduce-pilot experiment=pilot task=dummy_binary solver=none state_model=baseline storage=local tracking=local`と自動テストで確認する。実研究の研究質問、仮説、主要比較、test評価の回数はpilotから流用せず事前登録する。
