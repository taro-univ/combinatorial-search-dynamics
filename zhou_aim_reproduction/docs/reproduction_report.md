# Reproduction report

## Status

実装完了後のsmall profile実測値をここへ追記する。現時点で、直接N体系シミュレータ、観測量、可視化、数値検証scriptは実装済み。論文の完全なDMFT/1RSB自己無撞着解は対象外として分離している。

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
