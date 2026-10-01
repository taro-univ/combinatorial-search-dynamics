# Equation map: Zhou et al. の SK × AIM

## 一次資料

- Zhou et al., “Optimality and annealing path planning of dynamical analog solvers,” Physical Review Research 8, 033267 (2026). 以下「有限時間論文」。
- Zhou et al., “Phase analysis of Ising machines and their implications on optimization,” Physical Review Research 8, 033147 (2026); supplied arXiv manuscript `2507.08533v2.pdf`. 以下「相解析論文」。
- `夏期発表.pptx` の5–17、21ページは研究文脈の確認にのみ使用し、数式・規格化の根拠にはしていない。

## SK模型と decoded energy

論文は

\[
\min_{\sigma\in\{-1,+1\}^N}
-\frac12\sum_{ij}\sigma_iJ_{ij}\sigma_j
\]

を用いる（有限時間論文 Eq. (1)、相解析論文 Eq. (1)）。結合は上三角の独立成分について

\[
J_{ij}\sim\mathcal N(0,J^2/N),\quad J_{ji}=J_{ij},\quad J_{ii}=0
\]

である（有限時間論文 Sec. II、相解析論文本文）。本実装の既定値は論文の無次元化に合わせて `J_scale = 1`、`xi = 1` とする。

decoded spin と、1スピン当たりの decoded energy は

\[
\sigma_i(t)=\operatorname{sgn}x_i(t),\qquad
E_{\rm dec}(t)=-\frac{1}{2N}\sigma(t)^\top J\sigma(t)
\]

である（有限時間論文 Sec. II、相解析論文 Eq. (48) および本文）。これは有限サイズの特定インスタンスに対する値であり、熱力学極限の基底エネルギー密度
\(E_\infty\simeq-0.7632\)（有限時間論文 Eq. (11)）とは区別する。論文で引用される有限サイズ平均の近似は

\[
E_0(N)\simeq E_\infty+0.701N^{-2/3}
\]

であり、個々のインスタンスの厳密基底値ではない。

## CIMの運動方程式

直接N体系のsoft-spin CIMは

\[
\frac{dx_i}{dt}=-x_i^3+a(t)x_i+\xi\sum_{j\ne i}J_{ij}x_j+h_i(t)+\zeta_i(t)
\]

である（有限時間論文 Eq. (2)、相解析論文 Eq. (3)）。項の対応は次のとおり。

- 局所ポテンシャル由来: \(-x_i^3+a(t)x_i\)。ポテンシャルは \(x_i^4/4-a x_i^2/2\)。
- 相互作用: \(\xi\sum_jJ_{ij}x_j\)。
- 外場: \(h_i(t)\)。通常の直接シミュレーションでは0、応答関数の推定時だけ微小摂動を加える。
- ノイズ: \(\zeta_i(t)\)。

白色ノイズの規格化は

\[
\langle\zeta_i(t)\rangle=0,\qquad
\langle\zeta_i(t)\zeta_j(t')\rangle=2T(t)\delta_{ij}\delta(t-t')
\]

である（有限時間論文 Sec. II、相解析論文本文）。Euler–Maruyama離散化は

\[
x_i^{n+1}=x_i^n+f_i(x^n,a_n)\Delta t
+\sqrt{2T_n\Delta t}\,\eta_i^n,qquad
\eta_i^n\sim\mathcal N(0,1)
\]

とする。有限時間論文 Fig. 1 は \(\Delta t=0.02\)、Appendix C は \(\Delta t=0.05\) を用いる。

## digCIMとCIMの違い

digCIMはsoft amplitudeではなく符号を相互作用へ渡す:

\[
\frac{dx_i}{dt}=a(t)\phi_B(x_i)+\xi\sum_{j\ne i}J_{ij}\operatorname{sgn}(x_j)+\zeta_i(t),
\]

\[
\phi_B(x)=\max(-B,\min(B,x)).
\]

有限時間論文 Appendix F Eq. (F1) と相解析論文の “Digitization” 節に対応する。CIMの cubic saturation \(-x_i^3\) はなく、相互作用は \(x_j\) ではなく \(\operatorname{sgn}x_j\) である。有限時間論文 Eq. (F1) は自己項に \(a\phi(x_i)\) を書き、相解析論文本文は bounded variable に対して \(ax_i\) と書く。本実装は両者が一致する範囲を明確にするため、各Euler step後にも `[-B,B]` へ射影する方式を設定で切り替え可能にする。

## 分布、相関、応答

時刻 \(t\) の瞬時soft-spin分布は

\[
P_t(x)=\frac1N\sum_{i=1}^N\delta(x_i(t)-x)
\]

の有限N推定とする。相解析論文では disorder/thermal average を含め
\(P(x)=[\langle\delta(x_i-x)\rangle]_{J}\) と定義する。実装では各試行の経験分布を保存し、複数seed・複数instanceの集約を別段階で行う。

DMFTの二時刻相関と応答は

\[
Q(t,t')=\langle x(t)x(t')\rangle,
\qquad
G(t,t')=\left\langle\frac{\partial x(t)}{\partial h(t')}\right\rangle
\]

である（有限時間論文 Eq. (3) 直後）。直接N体系では

\[
\widehat Q(t,t')=N^{-1}\sum_i x_i(t)x_i(t')
\]

を保存時刻上で計算する。\(G\) は同一instance・初期状態・ノイズ系列のpaired simulationを用い、Rademacher方向 \(v_i\) の微小外場を1 stepだけ加える trace estimator

\[
\widehat G(t,t')=\frac{1}{N\epsilon\Delta t}
\sum_i v_i[x_i^{(+)}(t)-x_i^{(0)}(t)]
\]

で推定する。これはDMFT全応答行列の対角平均の数値近似であり、有限差分幅 \(\epsilon\) 依存性を伴う。

DMFTにおける decoded energy の閉形式は

\[
E_{\rm dec}(t)=-\int_0^t
\frac{\partial\operatorname{sgn}x(t)}{\partial h(t')}
\langle\operatorname{sgn}x(t)x(t')\rangle\,dt'
\]

（有限時間論文 Eq. (4)）。今回はこの自己無撞着DMFTを完全には解かず、直接N体系で定義どおりのIsing energyを計算する。

## 符号反転率と原点密度

保存区間 \([t,t+\Delta\tau]\) の単位時間・1スピン当たり符号反転率を

\[
\widehat\mu(t)=\frac{\#\{i:\sigma_i(t+\Delta\tau)\ne\sigma_i(t)\}}
{N\Delta\tau}
\]

と定義する。論文の1-step zero-crossing rateとの対応を保つため、保存間隔とは別に積分stepごとの反転数も蓄積する。

有限時間論文 Eq. (6)、Appendix B Eq. (B8) は、小さい \(\Delta t\) と非零温度で

\[
\sqrt{\Delta t}\,\mu(t)\simeq
\sqrt{\frac{T}{\pi}}P_t(0)
\]

を与える。したがって比較量として
`sqrt_dt_mu = sqrt(dt) * flip_rate_per_time` と
`theory_mu = sqrt(T/pi) * P0` を保存する。これは近似式であり、ドリフト、有限N、KDE帯域幅、複数回zero crossingを無視する誤差を含む。

論文 Eq. (7) の実効凍結指標は

\[
\tau_{\rm budget}=\frac{1}{N\mu(t)}
\]

（同式の表記は \(\sqrt{\Delta t}\mu\) を介して同値）で、1スピン反転までの待ち時間が利用可能時間と同程度以上なら実効凍結とみなす。有限runでは概ね \(\mu_{\rm th}=1/(N\tau_{\rm remaining})\) を診断値とするが、係数は普遍ではない。

## 論文の図との対応

- 有限時間論文 Fig. 1: 固定 `a=0`, `T=1e-5`, `N=2000`, `dt=0.02` のCIM。energy、\(Q(t,t)\)、応答を比較。
- Fig. 2: \((a,T)\) のdecoded-energy contourとeffective-gap contour。gain-only Path 1 と temperature-only Path 2。
- Fig. 3: `T=0.01` 固定のgain annealingと、`a=2` 固定のtemperature annealingにおける分布変化。
- Fig. 4 / Appendix C: `N=10000`, `dt=0.05`, 20試行、最大 \(10^6\) step。temperatureは `2 -> 0.01`、gainは `-2 -> a_e`、比較は同step数。
- Fig. 9: `a_e=2` の単一試行におけるdecoded energyと \(P(0)\)。
- Fig. 10 / Appendix D: gain `-2 -> 2` at `T in {0.1,0.05,0.035}`、temperature `1 -> 0` at `a in {1,1.5,2}`、`N=10000` の数値的凍結点。
- 相解析論文 Fig. 2: CIMの \((a,T)\) 相図。
- 相解析論文 Fig. 4: CIM/clipCIM/simCIM/digCIMのzero-temperature decoded energy。シミュレーションは `N=10000`, 10000 step, `T=1e-5`。

## 実装上、一意に決まらない事項

1. 論文は初期条件の全詳細を図ごとには記載しない。既定は `Normal(0, initial_std^2)` とし、`initial_std` と初期化方式を設定へ分離する。
2. `sgn(0)` の扱いは明記されない。実装では直前の符号を保持し、初期時点で厳密に0なら+1とする。
3. \(P_t(0)\) の数値推定法・帯域幅は一意でない。Gaussian KDE（Scott帯域幅×係数）と原点対称binの両方を保存する。
4. 有限時間論文 Fig. 2 の1RSB energy/P(0) contourを再現するには非線形自己無撞着方程式の解法と枝選択が必要で、直接N体系の定常推定とは同一ではない。本実装のphase mapはまず有限時間・有限Nの直接シミュレーション推定として明記する。
5. digCIMのclippingをdrift入力だけに使うか、各step後の状態へも射影するかは論文間の表記だけでは完全に一意でない。既定はbounded machineとしてstep後射影を行い、`project_after_step` で感度分析可能にする。
6. `T=0` では確率ノイズによる Eq. (6) の関係は適用できない。決定論的な符号反転は別に観測する。
7. 理論 \(E_\infty\) はdisorder averageの熱力学極限であり、有限Nのresidual計算には小規模では厳密列挙値、大規模では有限サイズ平均近似を別ラベルで使う。
