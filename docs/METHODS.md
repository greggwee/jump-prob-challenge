# Methods: a complete mathematical walkthrough

*Lecture notes for the Probability Cup forecasting system. Each section: the
math with full derivations, then the intuition, then where it lives in the code.*

Notation: $Y \in \{0,1\}$ is a binary outcome, $p = P(Y=1)$ a true probability,
$q$ a reported probability. $\sigma(x) = 1/(1+e^{-x})$ is the logistic function,
$\text{logit}(p) = \ln\frac{p}{1-p}$ its inverse.

---

## 0. The system end to end (read this first)

The sections below are organized by *mathematical topic*. This one is organized by
*what actually happens* when a fixture comes up, so the pieces have a place to live.
Bot A's job: turn "Spain vs Cape Verde, kicking off in 90 minutes" into an integer
1–99 for each of ~25 questions per match, and do it in a way that survives a
relative-Brier contest against a field that also reads the odds.

**The one-sentence thesis.** Build one coherent generative model of the match —
a pair of goal intensities $(\lambda_a, \lambda_b)$ that everything else descends
from — *anchor it to the betting market* so its absolute calibration is at least as
good as the crowd's, then spend a small, **gated** amount of deliberate
miscalibration only on the obscure markets where the crowd can't price and our
models genuinely deviate. Calibration is the floor; edge is the no-line markets.

**The pipeline, in order.**

1. **Standing data (fit once, point-in-time).** Elo ratings from every
   international result (§3); a decay-weighted Poisson goal model mapping Elo
   difference → expected goals (§2.2); a negative-binomial count model for
   corners/cards/SoT/offsides/fouls fit from a StatsBomb event stream (§5); club
   per-90 rates for player props from Understat + API-Football (§6.4). Every
   feature is the value *before* the match it predicts — the discipline that makes
   backtests honest (§3, point-in-time).

2. **A fixture arrives.** The scheduler wakes 6–25 min before a match's
   `opening_time` (= kickoff), pulls the live market list, and the parser
   (`submit/parser.py`) classifies each question into an *archetype* —
   `win`, `total_ge`, `ht_tied`, `team_corners_ge`, `player_sot_ge`, … — extracting
   subject team/player and thresholds. Archetypes group into four **families**
   (goal, count, player, timing) that share calibration and strategy treatment.

3. **The goal backbone.** Elo → $(\lambda_a, \lambda_b)$. From these we build a
   Dixon–Coles-corrected scoreline grid (§2.3) *and* run the v2 Monte-Carlo
   point-process simulator (§8) at the same intensities. The grid is fast and
   closed-form; the simulator captures state-dependent dynamics (trailing teams
   push, late-game cards) that the grid can't.

4. **The market anchor (the single most important step for not losing).** When
   bookmaker odds exist, de-vig each book (Shin, §4.2), combine books into one
   sharpness-weighted consensus (§10), invert the consensus 1X2 + totals back into
   *market* intensities $(\lambda_a^{\text{mkt}}, \lambda_b^{\text{mkt}})$ (§11),
   and geometric-blend in log-space: $\lambda^{\text{blend}} = \exp(0.85\ln\lambda^{\text{mkt}} + 0.15\ln\lambda^{\text{model}})$.
   Rebuilding the grid/sim on $\lambda^{\text{blend}}$ makes **every** downstream
   market — win, totals, halves, counts, props — inherit the market's information
   *coherently*, because they all descend from one intensity pair (§11).

5. **Read off every market.** From the (anchored) grid and simulator: 1X2 and
   totals directly; half-markets by Poisson thinning (§6.1); team counts from the
   NegBin pmfs (§5); "more corners than" comparisons by convolution/CDF (§5.2);
   player props by thinning the player's shrunk per-90 rate over a *minutes mixture*
   that respects Jensen (§6.2–6.3). Penalties/red cards use measured base rates.

6. **Calibration (§7).** Each family gets a Platt map $\sigma(a\,\text{logit}\,p+b)$
   fit on settled outcomes; when a family's forecasts barely vary, fall back to
   intercept-only. Anchored goal markets skip calibration — the market is already
   calibrated, re-Platting would double-count. Offsides carries its own map (its
   StatsBomb definition disagrees with the platform's "caught offside").

7. **Strategy: gated extremizing (§12).** The contest scores you *relative to the
   field*, so honest calibration on markets the field also prices earns ≈0. The
   extremizer amplifies only our *deviation from the field proxy*, per family, by a
   factor $k$ — and $k=1$ (honest) unless a gate (live or backtest) proves edge for
   that family. Anchored main markets are never extremized: there we *are* the
   field, so they're a pure calibrated floor.

8. **Submit and learn.** Round to an integer 1–99 (§1.3), batch-submit, log the
   blend/market/pure variants for CLV measurement. Daily, the learner refits
   calibration from new settlements and updates the edge gate. The whole loop is
   re-runnable and idempotent.

**Why this is statistically robust — the five load-bearing ideas.**

- **A strictly proper objective (§1).** Honesty minimizes expected Brier; any
  shading costs the *square* of the deviation. We only depart from honesty under
  the explicit, different rank objective of §12, and only where gated.
- **One coherent generative core.** Everything is a functional of $(\lambda_a,
  \lambda_b)$, so the 25 answers per match cannot contradict each other (P(win) and
  P(over 2.5) and P(team scores in 2H) are mutually consistent by construction).
  Patches like Dixon–Coles and frailty are corrections to *that* model, not
  separate models.
- **The market as a calibration floor.** Decades of efficiency research say the
  closing line is the best public forecast; anchoring to it removed our worst
  absolute blunders (the pure-Elo USA misprice, §11) and guarantees we don't lose
  *relative* ground to a sharper field on the main markets.
- **Uncertainty is propagated, not ignored.** The goal model's parameter
  covariance is carried through a sandwich estimator (§8.4) and averaged over MC
  draws, which automatically pulls thin-data probabilities back toward 50 — honest
  hedging worth Brier where the data is weak.
- **Every component had to earn its place (§9).** The decision rule was fixed
  *before* each experiment: keep a component iff it improves out-of-sample Brier on
  six point-in-time tournament backtests. Squad-quality and a GBM hybrid were
  *rejected* by this rule; their rejection is a finding about where the residual
  information lives (markets, lineups, club form), not a failure.

The honest current limitation is the **field proxy** (§12.5): we cannot see the
crowd, so we estimate it. That single approximation gates how much edge we dare
spend, and improving it is the highest-value open problem. §13 closes the loop with
what the *settled* results say about where this whole machine is actually winning
and losing.

---

## 1. The scoring rule: why honesty is mathematically optimal

### 1.1 The Brier score and strict propriety

You report $q$; the outcome is $Y$. Your loss is $S(q, Y) = (q - Y)^2$.
Your *expected* loss when the true probability is $p$:

$$E[S(q,Y)] = p\,(q-1)^2 + (1-p)\,q^2$$

Expand both terms:

$$= p(q^2 - 2q + 1) + (1-p)q^2 = pq^2 - 2pq + p + q^2 - pq^2$$

$$= q^2 - 2pq + p$$

Minimize over $q$: $\frac{d}{dq}\left[q^2 - 2pq + p\right] = 2q - 2p = 0
\Rightarrow q^\* = p$, and $\frac{d^2}{dq^2} = 2 > 0$ confirms a minimum.

There is a cleaner way to see *how much* a lie costs. Evaluate the expected
loss at $q = p$:

$$E[S(p,Y)] = p^2 - 2p^2 + p = p(1-p)$$

and subtract from the general expression:

$$E[S(q,Y)] - E[S(p,Y)] = q^2 - 2pq + p - p + p^2 = (q-p)^2$$

**The regret of reporting $q$ instead of your true belief $p$ is exactly
$(q-p)^2$.** This is the whole game in one line: any deviation from honesty —
shading toward 50 to be safe, exaggerating toward 0/100 to look confident —
costs you the *square* of the deviation, in expectation, every time. That is
what "strictly proper scoring rule" means.

*Example.* You believe Brazil wins with $p = 0.78$. Report 78: expected Brier
$0.78 \times 0.22 = 0.1716$. Report 90 instead: regret $(0.90-0.78)^2 = 0.0144$,
expected Brier $0.1860$. Doesn't sound like much — but the competition has ~700
questions, and edges per question of ±0.01 decide the leaderboard.

### 1.2 Relative scoring changes nothing (for expected points)

Points per question $= \bar{B}_{\text{field}} - B_{\text{you}}$. Take
expectations: $E[\text{points}] = E[\bar{B}_{\text{field}}] - E[B_{\text{you}}]$.
The first term does not depend on your report, so maximizing expected points
$\equiv$ minimizing your own expected Brier $\equiv$ honesty, by §1.1. (For a
*rank* objective like "finish first," variance matters and Lichtendahl–Winkler
show extremizing can pay — but that's a different objective, and with the
account-level prediction limit we play expected points.)

### 1.3 Integer reports

The API takes integers 1–99. Rounding $p$ to the nearest integer percent costs
at most $(0.005)^2 = 2.5\times10^{-5}$ regret per question — negligible. The
1/99 clip means even certainty costs $(0.01)^2 = 10^{-4}$ when right, and saves
$1 - 0.98 = 0.0199$ Brier when "certain" events fail — a cheap insurance the
platform enforces for us.

---

## 2. Modeling goals: the Poisson backbone

### 2.1 Why Poisson? The law of rare events, derived

Split the match into $n$ tiny intervals. In each, a goal happens with
probability $\lambda/n$, independently (one team, for now). The number of goals
is Binomial$(n, \lambda/n)$:

$$P(X = k) = \binom{n}{k}\left(\frac{\lambda}{n}\right)^k
\left(1 - \frac{\lambda}{n}\right)^{n-k}$$

Take $n \to \infty$ with $k$ fixed. Write out the binomial coefficient:

$$\binom{n}{k}\frac{1}{n^k} = \frac{n(n-1)\cdots(n-k+1)}{k!\,n^k}
= \frac{1}{k!}\prod_{j=0}^{k-1}\frac{n-j}{n} \longrightarrow \frac{1}{k!}$$

since each factor $\to 1$. Meanwhile

$$\left(1-\frac{\lambda}{n}\right)^{n-k}
= \left(1-\frac{\lambda}{n}\right)^{n}\left(1-\frac{\lambda}{n}\right)^{-k}
\longrightarrow e^{-\lambda}\cdot 1$$

using $\lim_{n\to\infty}(1 - \lambda/n)^n = e^{-\lambda}$. Therefore

$$P(X=k) \to \frac{\lambda^k e^{-\lambda}}{k!}$$

**Intuition:** goals are the canonical "many chances, each rarely converted"
process. ~90 minutes, a goal in any given minute with probability ~1.5%.
That's exactly the regime where the binomial converges to Poisson. (The v2
simulator in §8 literally implements the pre-limit version: 90 Bernoulli/Poisson
slots.)

### 2.2 The log-linear intensity model and its MLE

For each side $s$ of each match, we model goals as Poisson with

$$\log \lambda_s = \beta_0 + \beta_1 \cdot \frac{\text{Elo}_s - \text{Elo}_{\text{opp}}}{100} + \beta_2 \cdot \text{home-ground}_s$$

Write $x_i$ for the covariate vector of observation $i$, $\lambda_i = e^{x_i^\top\beta}$.
The weighted log-likelihood (weights $w_i$ in §2.4) is, dropping the
$-\ln y_i!$ constant:

$$\ell(\beta) = \sum_i w_i\left[y_i\, x_i^\top\beta - e^{x_i^\top\beta}\right]$$

Differentiate w.r.t. $\beta$, using $\frac{\partial}{\partial\beta} e^{x^\top\beta} = e^{x^\top\beta} x$:

$$\nabla\ell = \sum_i w_i (y_i - \lambda_i)\, x_i$$

This is the **score equation**: at the optimum, weighted residuals are
orthogonal to every covariate — the model is "unbiased along each feature
direction." Differentiate again:

$$\nabla^2 \ell = -\sum_i w_i\, \lambda_i\, x_i x_i^\top$$

This is negative definite whenever the $x_i$ span the space (each term is a
negative-scaled outer product), so $\ell$ is strictly concave: **the MLE exists,
is unique, and BFGS finds it**. That's why `fit_goal_model` is trustworthy.
A synthetic-recovery test (generate from known $\beta$, refit, compare)
verifies the implementation rather than the math.

*Why Elo as the covariate?* National teams play ~10 matches/year — too few to
estimate per-team attack/defence freely. Elo (next section) compresses a
century of results into one number per team, and the *difference* is the
sufficient statistic for relative strength. We measured $\beta_1 \approx 0.35$:
a 100-point Elo gap multiplies your expected goals by $e^{0.35} \approx 1.42$
and divides the opponent's by the same.

### 2.3 Dixon–Coles: fixing the low-score cells

Empirically, 0-0, 1-0, 0-1, 1-1 occur more often than independent Poissons
predict. Dixon & Coles multiply the joint pmf by a correction $\tau$:

$$P(x,y) = \tau_{x,y}\cdot\frac{\lambda^x e^{-\lambda}}{x!}\cdot\frac{\mu^y e^{-\mu}}{y!},\qquad
\tau = \begin{cases}
1-\lambda\mu\rho & (0,0)\\
1+\mu\rho & (1,0)\\
1+\lambda\rho & (0,1)\\
1-\rho & (1,1)\\
1 & \text{else}
\end{cases}$$

With $\rho < 0$ (we estimate $\rho \approx -0.03$ to $-0.06$): the $(0,0)$ and
$(1,1)$ cells gain mass, $(1,0)$ and $(0,1)$ lose it — more draws, exactly the
observed deviation. The $\tau$ weights are built so the corrections
*approximately* cancel ($E[\tau] \approx 1$); we renormalize the grid
numerically to make it exact. $\rho$ is estimated by restricting the
likelihood to the four low-score cells (everywhere else $\tau = 1$ contributes
nothing): maximize $\sum_{\text{low}} w_i \ln \tau_{x_i,y_i}$.

**Intuition:** at 0-0 or 1-1 late, neither team risks much; at 1-0 the leader
defends and the trailer commits — score dynamics create dependence that
independent Poissons can't see. (Preview: the v2 simulator *generates* this
dependence mechanically from state-coupled hazards instead of patching cells —
§8.6.)

*Worked example.* $\lambda = 1.4$, $\mu = 1.1$, $\rho = -0.05$. Independent
$P(0,0) = e^{-1.4}e^{-1.1} = e^{-2.5} = 0.0821$. Corrected:
$\tau_{00} = 1 - (1.4)(1.1)(-0.05) = 1.077$, so $P(0,0) = 0.0884$ — the
goalless draw gains ~8% relative probability.

### 2.4 Time decay as a pseudo-likelihood

Old matches inform less. We weight match $i$, age $t_i$ days, by
$w_i = 0.5^{\,t_i/h}$ (half-life $h$). The weighted objective is not a true
likelihood — it's a **pseudo-likelihood**: consistent for the parameters of a
slowly drifting process, but its curvature no longer equals the information,
which is precisely why the uncertainty in §8.4 needs a sandwich, not just the
Hessian. We swept $h \in \{1, 2, 4\}$ years in backtest: flat (±0.0001 Brier).
Elo already encodes recency, so the residual decay barely matters; we fixed
$h = 730$ days.

---

## 3. Elo, and why it is logistic regression in disguise

The expected score with rating difference $d$ (plus 100 home advantage):

$$W_e = \frac{1}{1 + 10^{-d/400}} = \sigma\!\left(\frac{\ln 10}{400}\, d\right)$$

— the logistic function with slope $\ln 10/400 \approx 0.00576$ per Elo point.
The update after a match with result $W \in \{0, \tfrac12, 1\}$:

$$R \leftarrow R + K\,G\,(W - W_e)$$

Compare with stochastic gradient descent on the log-loss (cross-entropy)
$-[W\ln W_e + (1-W)\ln(1-W_e)]$ as a function of $R$: its gradient is
$-\frac{\ln 10}{400}(W - W_e)$. So the Elo update **is** online logistic
regression with learning rate $K \cdot 400/\ln 10$, where each team's rating is
a parameter updated one match at a time. The $K$ schedule (60 World Cup, 40
qualifiers, 20 friendlies) is a per-observation learning rate reflecting
informativeness; the goal-margin multiplier $G$ injects scoreline information
the binary loss would discard.

**Point-in-time discipline:** our Elo column for a match is the rating *before*
it — by construction, since updates apply after. This is what makes backtests
honest. (Validation: our table's top three — Spain, Argentina, France —
matches eloratings.net independently.)

---

## 4. From bookmaker odds to probabilities

### 4.1 The overround

Decimal odds $o_i$; raw inverses $\pi_i = 1/o_i$ sum to $B = \sum_i \pi_i > 1$
(the margin). All de-vig methods answer: which true $p_i$ (summing to 1)
generated these prices?

- **Basic:** $p_i = \pi_i / B$ — assumes proportional margin. Keeps the
  favourite–longshot bias.
- **Power:** $p_i = \pi_i^k$, solve the scalar $k > 1$ so $\sum_i \pi_i^k = 1$.
  Raising to a power $>1$ shrinks small numbers relatively more — longshots get
  shaved harder, the right direction for FLB.

### 4.2 Shin's method, derived

Shin (1993) models a fraction $z$ of bettors as *insiders* who know the
outcome. The bookmaker, balancing the books against this adverse selection,
sets prices satisfying (Shin's equilibrium condition):

$$\pi_i^2 = B\, p_i \left(z + (1-z)\, p_i\right)$$

Treat this as the definition and solve for $p_i$ — every step:

$$\frac{\pi_i^2}{B} = z\,p_i + (1-z)\,p_i^2
\;\;\Longleftrightarrow\;\;
(1-z)\,p_i^2 + z\,p_i - \frac{\pi_i^2}{B} = 0$$

Quadratic in $p_i$ with $a = (1-z)$, $b = z$, $c = -\pi_i^2/B$:

$$p_i = \frac{-z \pm \sqrt{z^2 + 4(1-z)\pi_i^2/B}}{2(1-z)}$$

Probabilities are nonnegative, so take the $+$ root:

$$\boxed{\;p_i = \frac{\sqrt{z^2 + 4(1-z)\,\pi_i^2/B}\; -\; z}{2(1-z)}\;}$$

Then $z$ is pinned by coherence: solve $\sum_i p_i(z) = 1$ numerically (Brent's
method on a bracketing interval — `devig_shin`).

**Sanity check the limit $z \to 0$:** the formula becomes
$p_i = \sqrt{4\pi_i^2/B}/2 = \pi_i/\sqrt{B}$, and
$\sum p_i = B/\sqrt{B} = \sqrt{B} = 1$ forces $B = 1$: with no insiders there
is no margin. The model says the overround *exists because of* informed money,
and the size of the observed $B$ reveals $z$.

**Why it fixes FLB:** for small $p_i$ the $z p_i$ term dominates the
$(1-z)p_i^2$ term inside the pricing equation, so longshot prices carry
proportionally more insurance markup — inverting removes more from them. Our
unit test constructs a synthetic biased book and asserts Shin restores the
favourite better than basic normalization.

### 4.3 Combining forecasts: the log opinion pool

Given model $p_m$ and market $p_k$, we blend in log-odds:

$$\text{logit}(p) = w\,\text{logit}(p_k) + (1-w)\,\text{logit}(p_m)$$

Equivalently $\frac{p}{1-p} = \left(\frac{p_k}{1-p_k}\right)^w
\left(\frac{p_m}{1-p_m}\right)^{1-w}$ — a geometric mean of odds. Why not a
plain average of probabilities? Two reasons. (i) *Symmetry*: blending 0.9 and
0.99 should behave like blending 0.1 and 0.01; logit space treats the two
tails identically, probability space doesn't. (ii) *Sharpness*: linear pools
are systematically underconfident (a known result — the average of two
calibrated forecasters is not calibrated, it's overdispersed); geometric pools
largely avoid this. The weight $w$ is never hand-set: we fit it on 84 real
fixtures against FiveThirtyEight's published forecasts (optimum ≈ 0.85, flat
0.65–1.0).

---

## 5. Count markets: negative binomial machinery

### 5.1 Overdispersion and the Gamma–Poisson mixture, fully derived

Corners per team-half have variance/mean ≈ 1.42 (measured); Poisson forces
var/mean = 1. Model the excess with a match-level random effect: let
$G \sim \text{Gamma}(k, \text{scale}=1/k)$, so $E[G]=1$, $\text{Var}(G)=1/k$,
density $f(g) = \frac{k^k}{\Gamma(k)} g^{k-1}e^{-kg}$, and $X \mid G \sim
\text{Poisson}(\mu G)$. Marginalize:

$$P(X=x) = \int_0^\infty \frac{e^{-\mu g}(\mu g)^x}{x!}\cdot
\frac{k^k}{\Gamma(k)}g^{k-1}e^{-kg}\,dg
= \frac{\mu^x k^k}{x!\,\Gamma(k)} \int_0^\infty g^{x+k-1} e^{-(\mu+k)g}\,dg$$

The integral is a Gamma integral: $\int_0^\infty g^{a-1}e^{-bg}dg = \Gamma(a)/b^a$
with $a = x+k$, $b = \mu+k$:

$$P(X=x) = \frac{\Gamma(x+k)}{x!\,\Gamma(k)}\cdot
\frac{\mu^x k^k}{(\mu+k)^{x+k}}
= \frac{\Gamma(x+k)}{x!\,\Gamma(k)}\left(\frac{k}{k+\mu}\right)^k
\left(\frac{\mu}{k+\mu}\right)^x$$

— the **negative binomial** pmf. Its variance, by the law of total variance:

$$\text{Var}(X) = \underbrace{E[\text{Var}(X|G)]}_{E[\mu G] = \mu}
+ \underbrace{\text{Var}(E[X|G])}_{\text{Var}(\mu G) = \mu^2/k}
= \mu + \frac{\mu^2}{k}$$

So $\text{var/mean} = 1 + \mu/k$, giving the method-of-moments estimator used
in `_dispersion_k`:

$$k = \frac{\mu^2}{\text{Var} - \mu}$$

*Example:* corners per team-half, $\mu = 2.20$, var/mean $= 1.42$ → Var $= 3.12$
→ $k = 4.84/0.92 \approx 5.3$. As $k \to \infty$, Var $\to \mu$: Poisson is the
limiting case (our offsides). **Intuition:** $G$ is "what kind of match this
turned out to be" — open or closed. The mixture says counts are Poisson *given*
the match character, and the character itself is random. This identity is also
why the v2 simulator's frailty term (a literal Gamma multiplier) reproduces
NegBin marginals exactly — same math, generative instead of analytic.

### 5.2 Comparison questions

"Will A have more corners than B at half time?" With independent pmfs
$f_A, f_B$ (conditional independence given intensities — a v1 simplification
that v2 relaxes):

$$P(A > B) = \sum_{a\ge1} f_A(a)\,F_B(a-1)$$

where $F_B$ is the cdf. Note these questions are typically settled "No" on
ties — $P(A>B) + P(B>A) + P(\text{tie}) = 1$, and for discrete counts the tie
mass is large: with corner means ~4.4 each, $P(\text{tie}) \approx 0.15$. This
is why both teams' "more corners" probabilities sit well below 0.5 — a trap
for intuition the model handles automatically.

### 5.3 Sums and thresholds

"4+ total cards": the pmf of a sum of independent counts is the convolution
$f_{A+B} = f_A * f_B$ (computed with `np.convolve`), then
$P(\text{sum} \ge t) = \sum_{s \ge t} f_{A+B}(s)$.

---

## 6. Player props: thinning, conjugacy, Jensen

### 6.1 Poisson thinning, proved

Player generates shots on target as Poisson with rate $r$ per minute over $M$
minutes; total $N \sim$ Poisson$(rM)$. More generally, take a Poisson($\lambda$)
stream and keep each event independently with probability $s$. The kept count
$N_2$:

$$P(N_2 = j) = \sum_{n \ge j} P(N=n)\binom{n}{j}s^j(1-s)^{n-j}
= \sum_{n\ge j}\frac{e^{-\lambda}\lambda^n}{n!}\cdot\frac{n!}{j!(n-j)!}s^j(1-s)^{n-j}$$

$$= \frac{e^{-\lambda}(s\lambda)^j}{j!}\sum_{n \ge j}\frac{[(1-s)\lambda]^{n-j}}{(n-j)!}
= \frac{e^{-\lambda}(s\lambda)^j}{j!}\,e^{(1-s)\lambda}
= \frac{e^{-s\lambda}(s\lambda)^j}{j!}$$

**Thinned Poisson is Poisson with the thinned rate.** This single theorem powers
the half-splits everywhere: if 55.6% of goals fall in the second half
(measured), a team with match intensity $\lambda$ has 2H goals
Poisson$(0.556\lambda)$, so

$$P(\text{scores in 2H}) = 1 - e^{-0.556\lambda}$$

### 6.2 Shrinkage = Gamma–Poisson conjugacy

A player's rate $r$ is unknown. Prior: $r \sim \text{Gamma}(a, b)$ (shape $a$,
rate $b$), density $\propto r^{a-1}e^{-br}$. Observe $y$ events in $m$ minutes,
likelihood $\propto r^y e^{-rm}$. Posterior:

$$p(r \mid y) \propto r^{a-1}e^{-br}\cdot r^y e^{-rm} = r^{(a+y)-1}e^{-(b+m)r}$$

— Gamma$(a+y,\, b+m)$, with posterior mean $\frac{a+y}{b+m}$. Choose the prior
as "$M_0$ pseudo-minutes at the population rate $r_0$": $a = r_0 M_0$, $b = M_0$:

$$\hat r = \frac{y + r_0 M_0}{m + M_0}$$

which is exactly the code's formula. A rookie with 90 career minutes is mostly
prior; Son Heung-min with 1,500+ minutes is mostly himself. The backtest chose
$M_0 = 270$ for shots on target and $540$ for goals — rarer events need heavier
shrinkage, which the data confirmed rather than assumed.

### 6.3 The Jensen correction for minutes

$P(\ge 1 \text{ event} \mid M) = f(M) = 1 - e^{-rM}$ with
$f''(M) = -r^2 e^{-rM} < 0$: **concave**. Jensen's inequality (proof for the
concave case: the tangent line at $E[M]$, $t(x) = f(E[M]) + f'(E[M])(x - E[M])$,
lies above $f$ everywhere by concavity, so
$E[f(M)] \le E[t(M)] = f(E[M])$). Therefore

$$E\left[1 - e^{-rM}\right] \;\le\; 1 - e^{-r\,E[M]}$$

Plugging in *expected* minutes **overstates** the probability. The correct
computation mixes over scenarios:
$P = \sum_s w_s\,(1 - e^{-r M_s})$ over {starts and finishes, subbed, cameo,
unused}. *Example:* $r = 0.02$/min, scenarios (0.8, 80'), (0.12, 25'), (0.08, 0').
Naive: $E[M] = 67$, $1 - e^{-1.34} = 0.738$. Correct:
$0.8(1-e^{-1.6}) + 0.12(1-e^{-0.5}) = 0.8(0.798) + 0.12(0.393) = 0.686$.
A five-point overstatement — material, and largest exactly for rotation risks.

### 6.4 Where the rates come from: the club-data layer

§6.2 needs a per-90 rate $r_0$ and a player-specific count $y$ over $m$ minutes.
National-team history is far too thin (a striker plays ~8 competitive internationals
a year), so the signal must come from **club** football. Two sources, joined on a
folded name key (`normalize/players.py`: accent/punctuation/order-invariant, with a
strict-subset tier for middle names and an ambiguity guard):

- **Understat** — top-5 leagues + Russian, with **xG, npxG, shots, key passes, xA**.
  Rich quality signal, but no shots-*on-target* and no breadth.
- **API-Football** (api-sports.io) — broad leagues Understat misses (Saudi, Qatar,
  Liga MX, MLS, Championship, Primeira, Eredivisie, Süper Lig) and, crucially,
  **`shots.on` directly** — the exact quantity the `player_sot_ge` prop needs.
  (FBref would have been the natural single source but is Cloudflare-walled to every
  HTTP client; the pivot to API-Football is documented in the plan.)

**The load-bearing number is coverage, not richness.** A player with no club join
falls back to the data-starved population prior — the §6.2 shrinkage collapses to
$\hat r \approx r_0$, i.e. we price every uncovered striker identically. That is the
"Kubo failure mode," and it is *exactly* where the settled review (§13) finds our
biggest player-prop bleeds. So the gating metric is the **coverage hole**: the
fraction of players appearing in actual prop markets with no rate. The census
(`scripts/broad_rates.py`) measured it at **44% with
Understat alone, cut to 16% after API-Football** — the difference is the leverage to
even *attempt* a calibrated player-prop number rather than a base-rate guess.

Two disciplines carry over from the rest of the system. **Shrinkage by minutes**
(§6.2) means a 477-minute fragmented season contributes its rate but is pulled hard
toward the prior — high-variance samples don't get to shout. And **point-in-time
honesty**: rates are last-completed-season club form, known before kickoff, never
contaminated by the tournament we're predicting.

---

## 7. Calibration

### 7.1 The Murphy decomposition (why calibration is identifiable)

Bin your forecasts into groups $g$ with average forecast $\bar q_g$, realized
frequency $\bar y_g$, sizes $n_g$, overall base rate $\bar y$. Then the mean
Brier score decomposes (add and subtract $\bar y_g$ inside the square and use
the fact that within-bin cross terms vanish because $\bar y_g$ is the bin mean):

$$\bar B = \underbrace{\frac{1}{N}\sum_g n_g(\bar q_g - \bar y_g)^2}_{\text{calibration (want 0)}}
\;-\; \underbrace{\frac{1}{N}\sum_g n_g(\bar y_g - \bar y)^2}_{\text{resolution (want big)}}
\;+\; \underbrace{\bar y(1-\bar y)}_{\text{uncertainty (fixed)}}$$

Calibration error is *reducible by post-processing*; resolution is not — you
can't manufacture discrimination after the fact, but you can stop lying about
your confidence. That's the entire case for a calibration layer.

### 7.2 Platt scaling

Fit $p' = \sigma(a\,\text{logit}(p) + b)$ by logistic MLE on settled outcomes.
Two parameters per archetype family: $a < 1$ compresses overconfidence,
$b$ shifts systematic bias. We found $a \approx 0.72$ for tournament
match-winners — Elo gaps overstate win probability at tournaments (group games
compress; the favourite plays risk-averse). One caution discovered in
verification: when an archetype's forecasts barely vary (every "2+ cards in
2H" sits near 0.63), the slope $a$ is unidentifiable and free fits go wild
(negative $a$); we fall back to intercept-only ($a \equiv 1$, fit $b$).

---

## 8. The v2 engine: point processes, conditioning traps, and sandwiches

### 8.1 Inhomogeneous Poisson processes and their likelihood

Let events arrive with a time-varying hazard $h(t)$ — here, per minute slot
$m$: $h(m) = \lambda \cdot \frac{e^{\alpha_{b(m)} + \beta_{s(m)}}}{Z}$, with
$\alpha$ a minute-bucket profile, $\beta$ a score-state multiplier, $Z$ a
normalizer. Discretize (as football minutes naturally do): slot counts are
independent Poisson$(h(m))$ given the path. The likelihood of observed slot
counts $k_m$:

$$L = \prod_m \frac{e^{-h(m)}h(m)^{k_m}}{k_m!}
\;\Rightarrow\;
\ln L = \sum_m \left[k_m \ln h(m) - h(m)\right] + \text{const}$$

— "sum of log-hazards at events, minus integrated hazard." This is the
discrete Aalen/Cox form, and **it remains valid when $h$ depends on the
realized history** (the filtration): at each instant, the hazard is a function
of what has already happened, and the product of conditional densities is the
joint density. This matters in §8.3.

### 8.2 The conditional-multinomial trick (for exogenous-state families)

Problem: each match-team has its own base rate (nuisance); we only want the
*shape* ($\alpha, \beta$). Condition on the total. If
$X_m \sim \text{Poisson}(\lambda_m)$ independent and $\Lambda = \sum \lambda_m$,
then for $\sum x_m = n$:

$$P(X = x \mid \Sigma X = n)
= \frac{\prod_m e^{-\lambda_m}\lambda_m^{x_m}/x_m!}{e^{-\Lambda}\Lambda^n/n!}
= \frac{n!}{\prod_m x_m!}\prod_m\left(\frac{\lambda_m}{\Lambda}\right)^{x_m}$$

— multinomial with cell probabilities $\lambda_m/\Lambda$, in which any
constant multiplying all $\lambda_m$ **cancels**. So writing
$\lambda_m = c_{\text{match,team}} \cdot e^{\alpha_{b(m)} + \beta_{s(m)}}$,
the conditional likelihood depends only on $(\alpha, \beta)$:

$$\ell_c(\alpha,\beta) = \sum_{\text{events}}\left[\alpha_{b(e)} + \beta_{s(e)}\right]
- \sum_{\text{match-teams}} n_{mt}\,\ln\!\left(\sum_m e^{\alpha_{b(m)}+\beta_{s(m,\text{path})}}\right)$$

Nine free parameters per family, no nuisances. Valid for corners, cards,
fouls, SoT, offsides — because *their* events don't move the score, so the
state path is exogenous to them.

### 8.3 The feedback trap: why goals need the full likelihood

For goals, the trick is **biased**, and our synthetic test caught it before
any real data did: recovered $\alpha$'s inflated monotonically with time
(fitted late-bucket +1.20 when truth was +0.50). Why? Conditioning on the
total requires the total to be *ancillary* for the shape — carrying no
information about where events fall. But each goal **creates the state that
follows it**: an early goal manufactures lead/trail exposure for all later
slots, a 90th-minute goal manufactures none. Trajectories with given totals
are therefore a *selected* set whose state-exposure profile depends on the
parameters themselves — the denominator $\sum_m e^{\alpha+\beta(\text{path})}$
treats the path as fixed when it is partly an outcome.

The fix follows from §8.1: the **unconditional** point-process likelihood with
the realized (filtration-adapted) path is exactly valid — we just can't cancel
the nuisance, so we supply it as an offset: per match-team,
$\ln h_m = o_{mt} + \mu + \alpha_{b(m)} + \beta_{s(m)}$ with
$o_{mt} = \ln \lambda_{\text{goal model}}$, and one global intercept $\mu$
absorbing units. Synthetic recovery then passes to ±0.08 on all parameters.

*Moral for the course: "condition on totals to kill nuisances" is a standard
move, and it silently fails for self-exciting processes. Always test
estimators on data generated from known parameters.*

### 8.4 Sandwich covariance for the posterior predictive

We want parameter uncertainty for the goal model to propagate into
probabilities. The decay-weighted MLE is an M-estimator: $\hat\beta$ solves
$\sum_i \psi_i(\beta) = 0$ with $\psi_i = w_i(y_i - \lambda_i)x_i$. Taylor-expand
the estimating equation around the truth $\beta_0$:

$$0 = \sum_i \psi_i(\hat\beta) \approx \sum_i \psi_i(\beta_0)
+ \left[\sum_i \nabla_\beta \psi_i\right](\hat\beta - \beta_0)$$

$$\Rightarrow\quad \hat\beta - \beta_0 \approx
H^{-1}\sum_i\psi_i(\beta_0),\qquad H = -\sum_i \nabla\psi_i = \sum_i w_i\lambda_i x_i x_i^\top$$

Take variances (independent observations):

$$\text{Var}(\hat\beta) \approx H^{-1}\,
\underbrace{\text{Var}\!\left(\sum_i\psi_i\right)}_{M = \sum_i w_i^2 (y_i-\lambda_i)^2 x_i x_i^\top}\,
H^{-1}$$

For a *true* likelihood with $w_i \equiv 1$, the information identity gives
$M = H$ and the sandwich collapses to the familiar $H^{-1}$. With decay
weights the identity fails ($w^2 \ne w$), hence the full sandwich. Each MC
draw-block then samples $\beta \sim N(\hat\beta, H^{-1}MH^{-1})$.

**Why this makes extreme probabilities less extreme:** the map
$\beta \mapsto P(\text{win})$ is nonlinear and bounded; near $p \approx 0.9$
it is concave in $\beta$, so by Jensen (§6.3) $E_\beta[p(\beta)] < p(\hat\beta)$.
Averaging over parameter draws automatically hedges thin-data fixtures — a
"78%" backed by little data reports as ~74%, and §1.1 says that honesty about
uncertainty is worth Brier.

### 8.5 Frailty: closing the variance budget

The simulator's state coupling already generates overdispersion endogenously
(random score paths ⇒ random integrated hazards ⇒ mixed Poisson). Measured:
coupling alone takes corners' var/mean from 1.00 to 1.17 vs an empirical 1.42.
The remainder is a Gamma frailty $G$ on the integrated hazard — by §5.1 this
adds $\mu^2/k$ of variance, so we solve $1 + \mu/k = $ (residual var/mean) for
$k$ per family, by pilot simulation. SoT needed none ($k=\infty$): coupling
explains its entire excess.

### 8.6 What coupling buys — and the transfer lesson

With state multipliers measured from 314 matches (trailing teams score ×1.33,
shoot on target ×1.67; one-goal leaders ×0.90; cards ×5.5 late), the simulator
*generates* the low-score dependence that Dixon–Coles patches by hand:
trailing sides push, equalizers arrive, draws and 1-1s exceed the independent
baseline. The ablation backtest then delivered the discipline: goal coupling
improves draw/win/timing Briers (+0.001 to +0.0007), but count-family coupling
*hurts* out-of-sample (SoT −0.018) — in-sample multipliers of that size don't
transfer across tournaments at these sample sizes, the same failure mode as
referee effects. Production config: **couple goals only**. A measured effect
is necessary but not sufficient; it must also *predict*.

---

## 9. Decision rules for components (the meta-method)

Every candidate component faced the same mechanical rule: *keep iff
out-of-sample Brier improves on six point-in-time tournament backtests.*

- **Squad quality**: unregularized fit pegged at the search bound on 48
  training matches and "improved" OOS Brier — a textbook overfit that a ridge
  prior ($b \sim N(0, 0.1^2)$, i.e. MAP with penalty $b^2/2\tau^2$) deflated
  to $b \approx \pm 0.01$ and a REJECT. Lesson: with one parameter and 48
  observations, the prior is doing the inference.
- **GBM hybrid**: with Elo and the DC probability already among its features,
  boosting found nothing left (standalone worse; expanding-OOS blend weight
  chose 0 in 4/5 tournaments). Groll et al.'s gains came from covariates we
  don't have (market values, age curves) — the method wasn't wrong, our
  feature set didn't feed it.

Both rejections are *findings*, not failures: they tell us where the residual
information actually lives (markets, lineups, club form) and they were cheap
because the decision rule was fixed before the experiment ran.

---

## 10. Wisdom of crowds: the multi-book consensus

§4 de-vigged *one* book. Live, many books quote the same match. How to combine
them into one consensus probability? Each book $b$ gives a de-vigged estimate
$p_b$ of the same true $p$; model $p_b = p + \varepsilon_b$,
$\text{Var}(\varepsilon_b) = \sigma_b^2$. We want the weights $w_b$
($\sum_b w_b = 1$) minimizing the variance of $\hat p = \sum_b w_b p_b$.

### 10.1 Inverse-variance weighting, derived

Assume the book errors are uncorrelated (a simplification — see 10.2).
$\text{Var}(\hat p) = \sum_b w_b^2 \sigma_b^2$. Minimize subject to
$\sum_b w_b = 1$ via Lagrange:

$$\mathcal{L} = \sum_b w_b^2 \sigma_b^2 - \lambda\Big(\sum_b w_b - 1\Big),
\qquad \frac{\partial\mathcal L}{\partial w_b} = 2 w_b \sigma_b^2 - \lambda = 0
\;\Rightarrow\; w_b = \frac{\lambda}{2\sigma_b^2}$$

Normalizing, **$w_b \propto 1/\sigma_b^2$** — weight each book by its precision.
This is the same minimum-variance result behind the precision-weighted ensemble
in §5.3 of the plan; it recurs whenever you average noisy estimates of one
quantity.

We never observe $\sigma_b$. But the **bookmaker margin (overround) is a proxy**:
a sharp book runs a 2% margin with high limits and professional money; a soft
book runs 8% with low limits. Lower margin ⇒ lower $\sigma_b$. So
$w_b \propto 1/(\text{overround}_b + \epsilon)$ approximates inverse-variance
weighting, with $\epsilon$ stopping a zero-margin exchange from grabbing
infinite weight. A small named-book multiplier nudges venues known to be sharp
beyond what their margin shows. (`models/devig.py::consensus`.)

### 10.2 Why arithmetic mean here, geometric (logit) mean for model-vs-market

The pool is in **probability space**, not logit space — the opposite of §4.3.
The distinction is real: books are *correlated copies of one quantity* (most
shade the same sharp line), so you are averaging noisy measurements, and the
minimum-variance combiner is the arithmetic mean (10.1). Logit/geometric pooling
(§4.3) is for combining *diverse, independent* information sources — our model
versus the market — where a linear pool would be underconfident. Same verb
("average"), different geometry, chosen by what the inputs *are*.

### 10.3 Prediction markets as pseudo-books

A Kalshi/Polymarket mid-price is just another implied probability: feed it as a
book with odds $1/p$. Its overround is ~0, so 10.1 hands it high weight
automatically — correct, since a deep real-money market is sharp. Cap a thin
market by hand (its liquidity is its precision). The crowd and the bookmakers
thus enter one consensus, each weighted by sharpness.

---

## 11. The market anchor: blending intensities (Phase 7)

Decades of efficiency research say the closing line is the best public forecast
of a match. We blend it into the model at the level of the **goal intensities**,
so the anchor propagates coherently to *every* market (win, totals, halves,
counts via their $\lambda$ covariate, props via team scaling), not just 1X2.

Invert the consensus 1X2(+totals) to market intensities
$(\lambda_h^{\text{mkt}}, \lambda_a^{\text{mkt}})$ (the §2 inversion run on the
consensus), then geometric-blend with the model's:

$$\lambda^{\text{blend}} = \exp\big(w\ln\lambda^{\text{mkt}} + (1-w)\ln\lambda^{\text{model}}\big)$$

**Log space, not arithmetic** — intensities are positive scale parameters, and a
log blend is symmetric under "half the goals" / "twice the goals", the same
reason odds blend in logit space. Rebuild the Dixon–Coles grid (and the v2 sim)
on $\lambda^{\text{blend}}$ and read off all answers; coherence is automatic
because everything descends from one $\lambda$ pair. $w \approx 0.85$ was fit on
84 real fixtures against FiveThirtyEight (§4.3); books replace 538 live.

**The catch that motivates §12.** Anchoring fixes *absolute* calibration — it
removed our worst blunder (a pure-Elo model rating the USA 26% to win when the
market said ~50%, and they won). But the contest scores you *relative to the
field*, and the field reads odds too. The very thing that improves calibration
pulls you toward the crowd, where relative edge is structurally zero. That
tension is the subject of the final section.

---

## 12. Playing to win: relative scoring, rank objectives, extremizing

This is the strategic core, and it overturns §1's "just be honest." §1 is
correct for the objective it assumes (minimize your own expected Brier). But the
prize is **winner-take-most** — only rank #1 takes the fellowship — so the true
objective is $P(\text{finish \#1})$, and that changes everything.

### 12.1 Why variance helps when you want rank #1

Your cumulative score is $S \sim \mathcal N(\mu, \sigma^2)$ over many questions
(CLT). Let $M$ be the best of the other $K-1$ entrants' scores; you win iff
$S > M$. Treating $M$ as concentrated near a high threshold $\tau$ (only one of
many players is the best, so $\tau$ is large),

$$P(\text{win}) \approx \Phi\!\Big(\frac{\mu - \tau}{\sigma}\Big).$$

Now differentiate w.r.t. your own $\sigma$:

$$\frac{\partial}{\partial \sigma}\,\Phi\!\Big(\frac{\mu-\tau}{\sigma}\Big)
= \varphi\!\Big(\frac{\mu-\tau}{\sigma}\Big)\cdot\frac{\tau-\mu}{\sigma^2}.$$

Since you are (like everyone) **behind the projected winner**, $\tau > \mu$, so
this derivative is **positive**: more variance raises your chance of finishing
first. Honest calibration *minimizes* your variance-vs-field (that is what
"agree with the crowd where you can" does) — which is close to the worst thing
for $P(\text{rank \#1})$. The cost is that extremizing also lowers $\mu$ (you
become miscalibrated, raising expected Brier); the optimal variance trades the
two off, and the further behind you are ($\tau-\mu$ large), the more variance is
worth. This is the formal content of Lichtendahl & Winkler (2007): under
relative-rank scoring, proper rules stop being proper and the equilibrium is to
**extremize** — report more extreme than your honest belief.

### 12.2 Extremize the *deviation from the field*, not the raw probability

Naively pushing every probability toward 0/1, $p' = \sigma(k\,\text{logit}\,p)$,
is wrong here. On markets where you equal the field (the anchored main markets),
that is a symmetric gamble with **zero mean edge** — pure variance that also
raises your Brier. The right transform amplifies only where you *disagree*:

$$\text{logit}(p') = \text{logit}(p_{\text{field}})
+ k\big(\text{logit}(p_{\text{belief}}) - \text{logit}(p_{\text{field}})\big),
\qquad k \ge 1.$$

- $p_{\text{belief}} = p_{\text{field}}$ ⇒ $p' = p_{\text{field}}$: no gamble, a
  calibrated floor.
- $p_{\text{belief}}$ deviates ⇒ the deviation is amplified by $k$: variance
  concentrated exactly on your edge.

$k=1$ is honest reporting; $k>1$ is the aggression dial. (`models/strategy.py`.)

### 12.3 The field-correlation paradox, and where edge actually lives

Relative points per question are $(\bar B_{\text{field}} - B_{\text{you}})$. If
your forecast $\approx$ the field's, this is $\approx 0$ *no matter how
well-calibrated you are*. Since the field reads the same odds we anchor to, our
**main markets earn ~nothing** — the anchor is a defensive floor (it stops us
*losing* to a sharper field), not an offensive weapon. Points come from markets
the crowd **cannot** price off a line: team corners/cards/offsides, shots-on-
target comparisons, half-specific and timing markets, player props. There the
crowd guesses near base rates and our models genuinely deviate. So extremizing
must be **aimed at the no-line markets**, and the anchored main markets left as
the floor.

### 12.4 The danger, and evidence gating

Extremizing is **leverage on your edge** — it amplifies skill *and* error
symmetrically. It only helps rank if your deviations are genuinely better than
the field. Two cautions: our biggest pre-anchor deviation (the USA call) was
*wrong*, and on the no-line markets our edge over the field is **unvalidated**
(no field data, few settlements). So we gate: $k>1$ only on archetype families
where settled live results confirm we are beating the field, with enough sample;
otherwise $k=1$. We also raise $k$ late and when behind (12.1), and scale it by
the stage multiplier — a deviation in a knockout (2×) or final (3×) is worth
amplifying more.

### 12.5 The field proxy

We cannot see the crowd (the API hides it), so we proxy it. For a market with a
line, $p_{\text{field}} \approx$ the market, lightly dulled toward 50 (the crowd
is the sharp line plus noise). For a no-line market, the crowd clusters near the
archetype **base rate** and makes a *weaker* version of our adjustment:

$$\text{logit}(p_{\text{field}}) = \text{logit}(p_{\text{base}})
+ \rho\big(\text{logit}(p_{\text{belief}}) - \text{logit}(p_{\text{base}})\big),
\quad \rho \approx 0.5.$$

Our edge is then the part of the adjustment the crowd does *not* make,
$(1-\rho)(\text{logit }p_{\text{belief}} - \text{logit }p_{\text{base}})$, and
extremizing amplifies it. This is a coarse stand-in for a real field model — the
honest current limitation, and the next thing worth building.

*Synthesis: §1 says be honest to maximize expected points; §12 says a
winner-take-most race rewards variance on your genuine edge. The resolution is
not to abandon calibration but to use it as a floor and spend a gated, bounded
amount of variance precisely where — and only where — you can prove you beat the
crowd.*

---

## 13. What the settled results actually say (the empirical audit)

Theory is cheap; the contest is settling questions. As of the last review,
**253 settled, sign-determined questions across 26 matches**, mean Brier **0.2317**
(beats coin-flip 0.25). The public REST API exposes no field data (every
leaderboard/stats endpoint 500s — `audit_field_api.py`), so the per-question
analysis below estimates relative points with the *same* field proxy the live
extremizer uses (§12.5: belief shrunk toward 50 by per-family sharpness $\lambda$).


> **Reality check from the website (2026-06-19).** The platform front-end
> (`play.sportspredict.com`, behind login — the React app reads `/api/probability/*`,
> a different namespace than the public `/api/v1/*` we audited) reports gigabot1's
> **actual RBP = +520.41, rank #240 of 2,818** forecasters (top ~8.5%; the field is
> 317 bots + ~2,500 humans). And — correcting an earlier claim in this very note —
> the site **does expose the crowd per question**: `POST /api/probability/match-stats`
> returns, for each market, the crowd mean prediction (`prediction_average`), the
> outcome, the field mean Brier (`brier_average`), our Brier (`brier_user`), and the
> per-question relative score. **Scoring is confirmed relative Brier**:
> $\text{RBP}=\sum_q(B^{\text{field}}_q-B^{\text{you}}_q)\times100$, where
> $B^{\text{field}}$ is the *mean of individual forecasters' Briers* (not the Brier of
> the mean forecast). The per-question number in the collapsed history list is a
> different, **absolute** Brier×100 — conflating the two is what produced the wrong
> "only the aggregate is relative" claim earlier.
>
> **The crowd is NOT coin-flip — that earlier inference was wrong.** Direct example
> (CAN vs QAT, gigabot1): field $B=0.231$, us $0.222$, RBP $+9.56$. The crowd makes
> *sensible, varied* forecasts (Canada to win 71%, Qatar more fouls 56%, Qatar 5+
> corners 28%) — it is moderately calibrated, not clueless. Where it *does* default
> to ~50% is **player props it has no read on** (Jonathan David SoT: crowd 50%, us a
> confident-wrong 15% → **RBP −44**; Akram Afif SoT: crowd 42%, us a confident-right
> 9% → **RBP +19**). So our edge is real but **smaller per-question** than a
> coin-flip baseline implies, and player props are simultaneously our biggest upside
> *and* biggest blow-up risk — exactly the high-variance family the §6.4 club-data
> and SoT-calibration work targets. **Highest-value unlock: harvest `match-stats`
> across settled matches to replace the §12.5 *proxy* field with the measured crowd
> distribution**, then re-derive per-family edge and re-gate extremizing against the
> real field instead of a guess.

### 13.1 Where the points actually come from

| family | n | Brier | proxy RBP | reading |
|---|---|---|---|---|
| **timing** | 18 | 0.194 | **+53** | first-goal-half, HT-tied/leading: obscure, no line, crowd guesses near 50 — our thinning model genuinely deviates. Our best edge. |
| other (pen/red) | 13 | 0.217 | +12 | measured penalty/red base rates beat a clueless crowd. |
| player | 49 | 0.201 | −8 | *bimodal* — see below. |
| count | 102 | 0.239 | −8 | *bimodal* — corners/SoT win, offsides/fouls/cards bleed. |
| goal | 71 | 0.254 | −22 | **proxy artifact**: these are anchored, so the real field *is* the market and true relative damage on upsets ≈ 0 (the CLV ledger confirms the anchor tracks the close). Read as ≈floor, not a −22 loss. |

The family averages hide the real structure, which is **bimodal within count and
player**:

- **Genuine edge (extremize candidates if the gate confirms):**
  `team_corners_ge` (+36), `sot_gt` (+33), `corners_gt` (+26), `team_sot_ge` (+24),
  `first_goal_half` (+18), `ht_tied`/`ht_leading` (+28 combined), `player_scores`
  (+30). These beat even the assumption-light base-rate baseline by +0.05 to +0.21
  Brier — real skill, not proxy luck.
- **Negative-value markets (our model is worse than *guessing the base rate*):**
  `offsides_ge` (−51 RBP, −0.023 vs base over n=22), `fouls_gt` (−41, −0.030 over
  n=20), `team_cards_ge` (−0.187, n=2). Offsides and fouls are near-random at the
  team-half level; the model spends confidence it hasn't earned. **Action: shrink
  these hard toward the base rate** (cap their deviation), the single biggest
  available point recovery.

### 13.2 The crowd asymmetry, quantified

Per-question RBP is **left-skewed (skew −1.45)**: best single call +10.4, worst
−28.4 — a **2.7× downside/upside ratio**. This is the §1 algebra made physical:
relative points $= B_{\text{field}} - B_{\text{you}}$, and since $B_{\text{you}}\in
[0,0.98]$, a confident-wrong call ($B\to0.98$) loses almost a full point while a
confident-right call only earns back $B_{\text{field}}$ — and on easy questions the
field is right too, so the upside is small. **Being wrong while the crowd is right
is the only thing that really hurts; being right while the crowd is right earns
nothing.**

The confidence buckets show the trap directly:

| our lean $|p-0.5|$ | n | hit-rate | RBP / question |
|---|---|---|---|
| 0.0–0.1 | 88 | 55% | +0.09 |
| **0.1–0.2** | 84 | 63% | **+0.67** ← the sweet spot |
| 0.2–0.35 | 56 | 66% | −0.15 |
| **0.35–0.5** | 25 | **76%** | **−1.20** ← hits often, still loses points |

Our *most confident* bucket hits 76% of the time and **still bleeds −1.2 pts/q**,
because the 24% of misses cost ~−27 each and the 76% of hits only save a few. This
is decisive evidence **for** the gated, deviation-only extremizing of §12 and
**against** naive confidence: the points live in moderate, well-aimed leans, not in
maximal conviction. Net across everything: 157 questions earn (+801), 96 lose
(−775), for +26 — positive, but thin, and entirely dependent on *not* taking the
catastrophic confident-wrong calls.

### 13.3 The one systematic bias worth fixing now

Every one of our worst single bleeds is the **same shape**: we predicted a
shots-on-target / "both teams SoT" / player-SoT event *unlikely* and it happened
(Kubo p7→YES −28, both-teams-SoT p9→YES −28, Díaz p10→YES −28, Xhaka p11→YES −27).
`player_sot_ge` realizes **40%** but we predict a mean of **29%** — a −0.11
calibration bias toward under-calling shots on target. (The `both_teams_sot` cases
were a parser bug, since fixed.) Two fixes converge on it: (i) the §6.4 club-data
layer now supplies **direct `shots.on` rates** for exactly these players, which
should lift the under-prediction; (ii) a calibration intercept correction on the SoT
families. This is the clearest line from "the data says we're wrong here" to "here
is the mechanism that fixes it" — and the reason the player-data work was worth
doing.
