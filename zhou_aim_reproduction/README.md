# Zhou SK × AIM reproduction

Zhouらの2論文を一次資料として、Sherrington–Kirkpatrick模型上のAnalog Ising Machineを有限Nで直接シミュレーションし、探索の進行・停滞・凍結を診断する独立実験です。状態依存feedback controlはまだ含みません。スケジュールは将来差し替えられる独立interfaceです。

数式と論文中の式・図の対応は [`docs/equation_map.md`](docs/equation_map.md)、実測結果と未再現事項は [`docs/reproduction_report.md`](docs/reproduction_report.md) に記録します。

digCIMの不一致診断は、81試行と厳密列挙対照を含みます。N=2000・a=−1の冷却対照は平均energy −0.7538となり、固定低温の−0.6855から改善しました。`compare_steady_energy_by_gain.py` の「steady」は後半の時間平均を指し、平衡到達を保証しません。profile名が `full` でも、このscriptのstep数は `--burn-steps` と `--sample-steps` で決まります。

## セットアップ

リポジトリrootで次を実行します。独立したsubprojectなので、既存のナップサック実験環境を変更しません。

```bash
UV_CACHE_DIR=/tmp/zhou-aim-uv uv sync --project zhou_aim_reproduction --extra dev
```

## 最短の再現手順

まずCPUだけで終わるsmoke testを実行します。同一seedの再生成、instance/resultの再読込、厳密列挙とのenergy整合、paired responseまで検査し、図とGIFも生成します。

```bash
UV_CACHE_DIR=/tmp/zhou-aim-uv uv run --project zhou_aim_reproduction \
  python zhou_aim_reproduction/scripts/run_smoke_test.py --profile small
```

次に同じinstance・run seed・積分step数でtemperature annealingとgain annealingを比較します。

```bash
UV_CACHE_DIR=/tmp/zhou-aim-uv uv run --project zhou_aim_reproduction \
  python zhou_aim_reproduction/scripts/reproduce_dynamics.py --profile small --model-kind both
```

有限時間・有限Nの直接シミュレーションによる \((a,T)\) mapを作ります。これは論文Fig. 2の1RSB等高線そのものではなく、同じ観測量の数値推定です。

```bash
UV_CACHE_DIR=/tmp/zhou-aim-uv uv run --project zhou_aim_reproduction \
  python zhou_aim_reproduction/scripts/reproduce_phase_map.py --profile small --model-kind cim
```

時間刻み、N、seed、初期条件、KDE帯域幅、histogram bin幅、flip windowへの依存性を確認します。

```bash
UV_CACHE_DIR=/tmp/zhou-aim-uv uv run --project zhou_aim_reproduction \
  python zhou_aim_reproduction/scripts/run_validation.py --profile small
```

digCIMの不一致を切り分ける実験は次で実行します。81試行と16スピン系の厳密列挙対照を行い、時間刻みは物理時間を固定し、共通のBrownian増分を使って比較します。固定低温と、高温から線形冷却して最後の20%を低温で保持する条件を、同じinstance・初期値・noiseで比較します。N=2000を含むCPU実験なので数分以上かかります。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --project zhou_aim_reproduction \
  python zhou_aim_reproduction/scripts/diagnose_digcim.py
```

出力は `results/digcim_diagnosis/<run-id>/` の `runs.csv`、`summary.csv`、`manifest.json`、各試行のtrajectory、`figures/diagnosis.png` です。既定のサイズ比較は各Nにつき1個のinstance・3初期seedを用いる予備検証です。disorder平均を精密に評価する場合は `--instance-seeds 11 29 47 ...` でinstanceを増やします。有限サイズの参照線はensembleの基底energyの近似であり、特定instanceの厳密最適値ではありません。

単体テストは次です。

```bash
UV_CACHE_DIR=/tmp/zhou-aim-uv uv run --project zhou_aim_reproduction pytest zhou_aim_reproduction/tests
```

各scriptは既存結果を上書きせず、`results/<experiment>/<UTC timestamp>/` に新しいrun directoryを作ります。固定名が必要なら `--run-id NAME` を指定しますが、同名が存在すると停止します。

## 実装したモデル

CIMはEuler–Maruyama法で

\[
x_i^{n+1}=x_i^n+\left[-(x_i^n)^3+a_nx_i^n+\xi\sum_jJ_{ij}x_j^n\right]\Delta t
+\sqrt{2T_n\Delta t}\,\eta_i^n
\]

を積分します。digCIMはcubic項を除き、相互作用を \(J\operatorname{sgn}(x)\) に置き換えます。既定では各step後に `[-B,B]` へ射影します。この選択は `project_after_step` で感度分析できます。

実装済みscheduleは固定 \((a,T)\)、temperature-only、gain-only、任意のpiecewise-linear \((a,T)\) pathです。interfaceは時刻だけを受け取るopen-loop設計で、状態依存feedbackは意図的に未実装です。

## 保存する観測量

- soft spin `x` と decoded spin `sign(x)`
- decoded energy density、best-so-far、基準値からのresidual
- Gaussian KDEとsymmetric binによる `P_t(0)`、KDE帯域幅
- 単位時間・1スピン当たりflip rate、累積flip数
- `sqrt(dt) * mu` と論文近似 `sqrt(T/pi) * P_t(0)`
- amplitude平均・分散、near-zero fraction
- 初期状態・best-so-far状態とのspin overlap
- 保存時刻上の `Q(t,t')`
- 共通noise paired simulationによる有限差分 `G(t,t')`
- seed、全parameter、日時、Git commit/dirty flag、Python/package環境

## 可視化

- energy・活動度・parameterの同期時系列
- soft amplitudeとdecoded signのheatmap
- `P_t(x)` の時間発展
- 小規模系の全状態energy分布、基底状態からのHamming距離、訪問状態、best更新
- soft-spin高次元軌道のPCA射影（真の2次元potential地形ではない）
- \((a,T)\) 上のenergy、`P(0)`、flip rate、annealing path、数値的凍結点
- 代表試行のGIF animation

既存`.npz`から図を再生成する例です。

```bash
UV_CACHE_DIR=/tmp/zhou-aim-uv uv run --project zhou_aim_reproduction \
  python zhou_aim_reproduction/scripts/visualize_trajectory.py \
  RESULT.npz --instance INSTANCE.npz --output-dir /tmp/zhou-aim-figures --animate
```

## 計算量profile

- `small`: CI/smokeと定性的検証向け。論文の有限サイズ・試行数は再現しない。
- `medium`: 数百spin、より長いtrajectoryと細かいphase grid。
- `full`: 論文の `N=2000`、最大 `10^6` stepを含む重い設定。dense SKの1 stepは \(O(N^2)\) なので、実行時間とmemoryを確認してから使う。

`full`でも論文Appendix Cの `N=10000 × 10^6 step × 20 trials` をそのままCPU実行する設定にはしていません。その規模はdense matrix-vector積だけで極めて大きく、GPU/compiled batch実装と計算資源の明示が必要です。

## ディレクトリ

```text
configs/                    small / medium / full
src/zhou_aim/               simulator, observables, exact solver, plots
scripts/                    executable experiments
tests/                      deterministic CPU tests
docs/equation_map.md        equations, normalization, assumptions
docs/reproduction_report.md reproduced claims and limitations
results/                    generated run directories (Git ignored by root policy)
```
