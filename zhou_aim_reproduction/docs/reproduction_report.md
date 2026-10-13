# Reproduction report

## Status

直接N体系シミュレータ、観測量、可視化、数値検証scriptは実装済み。2026-10-02にdigCIMの不一致診断81試行と、小規模系の厳密列挙による3試行の対照を完了した。論文の完全なDMFT/1RSB自己無撞着解は対象外として分離している。

## digCIMの不一致診断（2026-10-02）

### 結論

今回の主な差は、静的理論の到達可能な基底energyと、最初から低温で走らせた有限時間ダイナミクスを比較していたことである。低温開始は局所解への凍結を起こす。同じN・gain・step数でもtemperature annealingで大幅に改善した。時間刻みの影響も確認できたが、刻みを小さくするだけでは基底値へ近づかなかった。有限サイズの基底energy補正だけでは差の大きさを説明できない。

相解析論文 [arXiv v2, Fig. 4(a)](https://arxiv.org/pdf/2507.08533v2#page=4) の黒線はゼロ温度の理論曲線であり、legendも `digCIM & E_SK (1RSB)` としている。灰色帯・点線はCIMの数値実験。本文は、静的理論で基底energyと一致してもダイナミクスが局所解で止まる場合を明記している。黒線を、固定低温・ランダム初期値の実行で必ず達する値とは解釈しない。

同論文の [Supplementary Material Fig. 13](https://arxiv.org/pdf/2507.08533v2#page=44) のdigCIM数値実験はN=10000、B=4、Δt=0.1、100000 steps、20試行、T=1から0へのannealingを使用する。今回の冷却対照はN=2000、B=1、Δt=0.02、物理時間2000なので、同図の定量再現とは呼ばない。

### 元の図の実行条件

`results/steady_gain_comparison/steady_cim_vs_digcim_n2000_t2000_20261002/manifest.json` とCSVを確認した。

- N=2000、J_scale=ξ=1、B=1、step後射影あり。
- Δt=0.02、burn=50000、sample=50000 steps。合計の物理時間は2000。
- T=10⁻⁵を開始時から固定。初期値は標準偏差0.1のGaussian。
- instance seedは11のみ、run seedは101、202、303。
- 図のenergyはburn以降の時間平均。`full.toml` の1000000 stepsはこのscriptで上書きされていた。
- エラー帯は3 runからの近似的な正規95%区間。論文のextremal rangeとは異なり、disorder全体の不確かさも表さない。

「steady」は後半の平均という集計名で、平衡到達の検査ではなかった。scriptの新しい図ではfinite-time tailと表記し、実際のΔt・物理時間・resolved config、tailのflip数を保存する。

### 温度経路の対照

同一instance、run seed、初期値、noise、a=−1、N=2000、Δt=0.02、100000 steps、B=1で比較した。冷却条件は前半80%でT=1→10⁻⁵を線形に下げ、残り20%は10⁻⁵に保持する。表は3 runの最終decoded energy平均と標本標準偏差。元の固定低温の後半平均は、各runの最終値とも一致していた。

| 条件 | 平均最終energy | run間SD |
| --- | ---: | ---: |
| 開始からT=10⁻⁵固定（元データ） | −0.685480 | 0.009537 |
| T=1から冷却して低温保持（新規計算） | −0.753828 | 0.002268 |
| N=2000の有限サイズ平均基底値の近似 | −0.758784 | — |

改善量は0.068348。残りの参照差は約0.004956であり、厳密な基底状態への到達は主張しない。参照値はensemble平均の近似で、seed=11の厳密最適値は分かっていない。

### 時間刻みと実行時間

N=512、a=−1、物理時間200、固定T=10⁻⁵を、同じ初期値とBrownian経路で比較した。細かい刻みのnoise増分を足して粗い刻みを生成し、刻み変更とnoise変更を分離した。

| Δt | steps | 3 runの平均最終energy | run間SD |
| ---: | ---: | ---: | ---: |
| 0.1 | 2000 | −0.666297 | 0.008028 |
| 0.02 | 10000 | −0.687494 | 0.010039 |
| 0.005 | 40000 | −0.664339 | 0.005721 |

到達する局所解は刻みに依存した。しかし小刻みほど基底値に近づく傾向は確認できず、刻みの収束も未確立。これはN=512の予備検証なので、N=2000での定量的な刻み誤差評価とは区別する。

N=512、Δt=0.02、a=−1で物理時間を200→2000にすると、固定低温の平均最終energyは−0.682735のまま、冷却条件は−0.723646→−0.740813へ改善した。N=2000の固定低温では200→2000で−0.679626→−0.685480と少し改善するため、実行時間の影響が全くないとは言わない。重要なのは低温で待つ時間と、探索可能な温度域に費やす時間を区別すること。

### 系のサイズ

N=128、512、2000それぞれで、同じseed番号の1 instance・3 runを物理時間200、a=−1、Δt=0.02で実行した。

| N | 固定低温の平均最終energy | 冷却ありの平均最終energy | 有限サイズ参照 |
| ---: | ---: | ---: | ---: |
| 128 | −0.693546 | −0.736564 | −0.735600 |
| 512 | −0.682735 | −0.723646 | −0.752247 |
| 2000 | −0.679626 | −0.736981 | −0.758784 |

有限サイズ近似E₀(N)=−0.7632+0.701N^(−2/3)では、N=2000と10000の参照差は0.002906。元のN=2000・a=−1の参照差0.073304より小さい。サイズが探索難度・凍結確率へ与える影響は別に残る。各Nのcoupling行列は異なり、1 instanceずつなので、表から厳密なsize scalingは推定しない。N=128で参照を下回る値も、ensemble近似が各instanceの下限ではないことと整合する。

### 凍結しても基底状態とは限らないことの直接確認

N=16、instance seed=11の全65536配置を列挙し、基底energy=−0.658819を求めた。a=−1、T=10⁻⁵、Δt=0.02、10000 stepsのdigCIMでは以下となった。

| run seed | 最終energy | 1スピン反転でenergyを下げられるスピン数 | 最後の物理時間38でのflip数 |
| ---: | ---: | ---: | ---: |
| 101 | −0.439011 | 0 | 0 |
| 202 | −0.658819 | 0 | 0 |
| 303 | −0.439011 | 0 | 0 |

符号が局所場と整合する静止状態でも、基底状態以外の局所解を含むことが直接確認できた。式による説明は `equation_map.md` に記載した。正gainでは自己増幅による固定も起き、N=512、a=0.5の固定低温対照では、1スピン反転でenergyを下げられるスピンが最終状態に平均4.49%残った。

### 実装検査と保存先

Jの対称性、対角0、上三角分散1/N、energyの1/(2N)、noiseの√(2TΔt)、cubic項の除去、符号を渡す相互作用、境界射影を照合した。N=512、2000の実測分散/期待分散はそれぞれ1.001114、1.001143。小規模対照では行列式でのenergyと独立した上三角辺和が一致した。今回の差を説明する規格化の誤りは確認しなかった。

新規実験は `results/digcim_diagnosis/digcim_diagnosis_20261002/` に保存した。

- `runs.csv`、`summary.csv`: 81診断試行の個別・集約値。
- `manifest.json`: seed、時間、設定、環境、Git provenance。
- `instance_n*_seed*.npz`、`trajectories/`: 行列と各trajectoryの再読込可能なデータ。
- `small_exact_checks.csv`、`small_exact_config.json`: 厳密列挙対照。
- `normalization_checks.json`: 結合の規格化チェック。
- `figures/diagnosis.png`: gain、Δt、N、実行時間の比較。エラー棒はrun間SD。

再実行はREADMEの `diagnose_digcim.py` コマンドを使用する。新規scriptの小規模実行、既存gain比較scriptの4条件実行、既存6テスト、変更scriptのruff check/formatを確認した。1 instance・3 seedという制限、全N=10000条件と1RSB数値解の未再現は残る。

## 再現判定の原則

- 論文と同じ主張を確認できた場合も、small finite-N runを論文の定量再現とは呼ばない。
- thermal-limit referenceとfinite-instance exact ground stateを分ける。
- temperature/gain比較は同一instance、run seed、積分step数、matrix-vector積数で行う。
- 不一致は規格化、有限サイズ、step幅、初期条件、run長、sample数の順に調べる。

## 実装済みだが次段階の項目

- 有限N直接シミュレーションによるphase mapは実装済み。
- DMFT Eq. (3) のcolored noise・memory kernelの自己無撞着時間発展、および1RSB equilibrium contourは未実装。
- 完全な論文Fig. 1–4再現には `N=2000–10000` と長時間・多数試行が必要で、small profileの後に計算資源を確認して段階的に実行する。

## FB制御へ向けた候補

入力候補はgain \(a(t)\)、noise temperature \(T(t)\)、必要ならcoupling strength \(\xi(t)\)。観測候補は短時間windowのflip rate \(\mu(t)\)、KDEによる \(P_t(0)\)、energy改善率、soft-spin fraction、amplitude varianceである。最初のfeedback仮説は、energyが停滞しかつ \(\mu\) または \(P_t(0)\) が閾値へ近づいたとき、gain増加を止めるかtemperatureを一時的に維持・増加してsoft populationを保つこと。ただし今回は実装しない。
