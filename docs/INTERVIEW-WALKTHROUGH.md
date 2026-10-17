# The Probability Cup system: an interview study guide

*A full walkthrough of the forecasting system that finished 67th of 4,013 entrants
(top 1.7%) and 16th of 498 bots in the Jump Trading Probability Cup at the 2026 World
Cup. It is written so you can rebuild every step from first principles. Each idea is
broken into its smallest pieces. Every variable is defined before it is used. Every
formula comes with the reason it is there. Every worked number was recomputed from this
repository's own code and data.*

The story has three acts:

1. **The objective.** What relative Brier scoring rewards. Where points can come from
   and where they can't.
2. **The group stage.** Anchor to the betting market and build the machinery that turns
   odds into probabilities. Then extend that one coherent model to the many questions
   no bookmaker quotes.
3. **Edge.** How you tested whether any deviation from a calibrated price actually
   paid, and how you then spent that edge only on the question types where it had been
   measured.

Part 4 reviews the results honestly. Part 5 is a drill of likely interview questions
with short answers. Part 6 is a one-page cheat sheet of numbers.

---

## Contents

- [Part 0 — Notation](#part-0--notation)
- [Part 1 — The objective: relative Brier](#part-1--the-objective-relative-brier)
- [Part 2 — Group stage: calibrate to the market, build the machinery](#part-2--group-stage-calibrate-to-the-market-build-the-machinery)
- [Part 3 — Testing the edge, then spending it where it exists](#part-3--testing-the-edge-then-spending-it-where-it-exists)
- [Part 4 — What happened, honestly](#part-4--what-happened-honestly)
- [Part 5 — Interview drill](#part-5--interview-drill)
- [Part 6 — Cheat sheet](#part-6--cheat-sheet)
- [Appendix — Where everything lives in the code](#appendix--where-everything-lives-in-the-code)

### Timeline at a glance

| dates (2026) | stage | what you built |
|---|---|---|
| Jun 11–14 | group stage, week 1 | Elo, Poisson/Dixon–Coles goal model, count models, parser, submit path; odds anchor (de-vig → implied intensities → blend); field model and extremizer; closing-line-value ledger; offline backtest gate |
| Jun 15–20 | group stage, week 2 | club-rate player data (coverage hole 44% → 16%); first parser bug fixes; **first harvest of the real crowd** (257 settled questions); confidence floor |
| Jun 28 – Jul 3 | round of 32 (2× stakes) | market-first rewrite; drop audit 56 → 0; player props anchored to book lines; crowd-proxy clamp; fouls and offsides pinned flat |
| Jul 4–11 | R16 and QF (2×) | new question templates built from existing pieces; crowd lift on player props; lineup overrides; endgame variance dial (off by default) |
| Jul 14–19 | SF, bronze (2×), final (3×) | final-round templates; staged lineup-conditional rules; stayed honest |

---

## Part 0 — Notation

You will be asked to define things precisely, so here is the notation used throughout.

**Probabilities and scoring**

| symbol | meaning |
|---|---|
| $j$ | index of a question (one binary market) |
| $y_j \in \lbrace 0,1 \rbrace$ | the realised outcome (1 = YES) |
| $p^{\star}_j$ | the true probability that $y_j = 1$ (unknown) |
| $p_j$ | your honest belief (your model's estimate of $p^{\star}_j$) |
| $q_j$ | the probability you actually submit |
| $q_{ij}$ | forecaster $i$'s submission on question $j$ |
| $N_j$ | number of forecasters on question $j$ |
| $\bar q_j = \frac{1}{N_j}\sum_i q_{ij}$ | the crowd's **consensus**, i.e. its mean forecast |
| $s_j^2 = \frac{1}{N_j}\sum_i (q_{ij}-\bar q_j)^2$ | the crowd's **dispersion**: how much forecasters disagree |
| $B_j = (q_j - y_j)^2$ | your Brier score on question $j$ |
| $F_j = \frac{1}{N_j}\sum_i (q_{ij}-y_j)^2$ | the **field average Brier**, which is the scoring baseline |
| $m_j \in \lbrace 1,2,3 \rbrace$ | stage multiplier: group 1×, knockouts 2×, final 3× |
| $\sigma(x) = 1/(1+e^{-x})$ | logistic function |
| $\operatorname{logit}(p) = \ln\frac{p}{1-p}$ | its inverse, the log-odds |

**Match model**

| symbol | meaning |
|---|---|
| $R_A, R_B$ | Elo ratings of teams A and B before the match |
| $d = (R_A - R_B)/100$ | Elo gap in units of 100 points |
| $\lambda_A, \lambda_B$ | goal intensities: expected goals for each team over 90 minutes |
| $\Lambda = \lambda_A + \lambda_B$ | total goal intensity of the match |
| $\beta_0, \beta_1, \beta_2$ | goal-model coefficients (intercept, Elo slope, home ground) |
| $\rho$ | Dixon–Coles low-score dependence parameter |
| $G[x,y]$ | the scoreline grid: $P(\text{A scores } x, \text{B scores } y)$ |
| $h_1 = 0.444,\; h_2 = 0.556$ | share of goals in the first and second half (measured on 47k goals) |

**Bookmaker odds**

| symbol | meaning |
|---|---|
| $o_i$ | decimal odds on outcome $i$ |
| $\pi_i = 1/o_i$ | raw implied probability |
| $S = \sum_i \pi_i$ | booksum ($S - 1$ is the overround, i.e. the book's margin) |
| $z$ | Shin's insider fraction |
| $w$ | market weight in the model/market blend ($w = 0.85$) |

**Counts and players**

| symbol | meaning |
|---|---|
| $\mu$ | mean of a count (corners, cards, shots on target, ...) |
| $k$ | negative-binomial dispersion; $k \to \infty$ gives Poisson |
| $r$ | a player's event rate per minute |
| $M$ | minutes played; $M_0$ = prior pseudo-minutes for shrinkage |

**Strategy**

| symbol | meaning |
|---|---|
| $f$ | estimated field probability (what we think the crowd will submit) |
| $\lambda_f$ | field sharpness for a question family (0 = clueless crowd at 50%, 1 = crowd equals truth) |
| $\kappa$ | extremizing factor ($\kappa = 1$ is honest; the code calls this `k`) |
| $s$ | risk-gate shrink factor ($s < 1$ pulls toward a base rate) |

---

## Part 1 — The objective: relative Brier

### 1.1 Building block: the Brier score

You submit a probability $q \in [0,1]$. The outcome is $y \in \lbrace 0, 1 \rbrace$.
Your loss is

$$B = (q - y)^2 .$$

- $B = 0$ is perfect and $B = 1$ is maximally wrong. Lower is better.
- Example from the brief: $q = 0.75$ and $y = 1$ give $B = 0.25^2 = 0.0625$.
- Predicting 50 always scores $0.25$. That is the "coin flip" benchmark.

### 1.2 Building block: why honesty is optimal (strict propriety)

Suppose the true probability is $p^{\star}$. Your expected Brier for a report $q$ is

$$
\mathbb{E}[B] = p^{\star}(q-1)^2 + (1-p^{\star})q^2 .
$$

Expanding term by term:

$$
= p^{\star}q^2 - 2p^{\star}q + p^{\star} + q^2 - p^{\star}q^2 = q^2 - 2p^{\star}q + p^{\star}.
$$

Complete the square by adding and subtracting $p^{\star 2}$:

$$
\boxed{\;\mathbb{E}[B] = (q - p^{\star})^2 + p^{\star}(1-p^{\star})\;}
$$

Read this as two pieces:

- $p^{\star}(1-p^{\star})$ is **irreducible uncertainty**. Nothing you do changes it.
- $(q - p^{\star})^2$ is the **regret**, the penalty for reporting something other than
  the truth. It is zero only at $q = p^{\star}$, and it grows with the *square* of the
  deviation.

This is what "strictly proper" means: the unique minimiser of expected loss is the
truth. Shading toward 50 to be safe, or toward 0 and 100 to look confident, both cost
$(q-p^{\star})^2$ in expectation.

*Example.* You believe $p = 0.78$ and are right about that. Reporting 78 gives
expected Brier $0.78 \times 0.22 = 0.1716$. Reporting 90 adds regret
$(0.90-0.78)^2 = 0.0144$, for $0.1860$. That gap is small per question, but there were
about 1,169 settled questions.

### 1.3 Building block: the relative scoring rule

You are not paid your Brier. You are paid how much better you did than the field:

$$
\text{RBP}_j = 100 \cdot m_j \cdot \big(F_j - B_j\big),
\qquad
F_j = \frac{1}{N_j}\sum_{i=1}^{N_j}(q_{ij}-y_j)^2 .
$$

- $F_j$ is the **average of individual forecasters' Brier scores**. The platform's own
  per-question stats confirmed this, as opposed to the Brier of the average forecast.
  That distinction is the most important fact in the whole contest (§1.4).
- $m_j$ is the **stage multiplier**: 1× in the group stage, 2× in the knockouts, 3× in
  the final. It multiplies losses as well as gains.
- Your leaderboard score is $\sum_j \text{RBP}_j$.

### 1.4 The key identity: the field average is soft (Jensen)

Write each forecaster's error as their distance from the consensus plus the
consensus's distance from the outcome:

$$
q_{ij} - y_j = \underbrace{(q_{ij} - \bar q_j)}_{\text{disagreement}} + \underbrace{(\bar q_j - y_j)}_{\text{consensus error}} .
$$

Square and average over forecasters $i$:

$$
F_j = \frac{1}{N_j}\sum_i (q_{ij}-\bar q_j)^2 + 2(\bar q_j - y_j)\cdot\underbrace{\frac{1}{N_j}\sum_i (q_{ij}-\bar q_j)}_{=\,0} + (\bar q_j - y_j)^2 .
$$

The cross term vanishes because deviations from a mean sum to zero. So

$$
\boxed{\;F_j = (\bar q_j - y_j)^2 + s_j^2\;}
$$

**The field's average Brier equals the Brier of the consensus forecast plus the crowd's
dispersion.** This is Jensen's inequality for the convex function
$x \mapsto (x-y)^2$: the average of the losses is at least the loss of the average, and
the gap is exactly the variance. You can also see it as the bias–variance decomposition,
applied across forecasters.

### 1.5 Expected points: where points can come from

Take expectations over the outcome $y_j \sim \text{Bernoulli}(p^{\star}_j)$. From §1.2,
$\mathbb{E}[(\bar q - y)^2] = (\bar q - p^{\star})^2 + p^{\star}(1-p^{\star})$, and the
same holds for your own report. The irreducible term $p^{\star}(1-p^{\star})$ appears in
both and cancels:

$$
\boxed{\;\frac{\mathbb{E}[\text{RBP}_j]}{100\,m_j} = \underbrace{s_j^2}_{\text{dispersion bonus}} + \underbrace{(\bar q_j - p^{\star}_j)^2}_{\text{crowd bias}} - \underbrace{(q_j - p^{\star}_j)^2}_{\text{your error}}\;}
$$

This one line tells you how to play the game.

| strategy | expected points per question (÷100m) |
|---|---|
| submit the consensus, $q = \bar q$ | $s^2$, **positive whatever the crowd's bias** |
| submit the truth, $q = p^{\star}$ | $s^2 + (\bar q - p^{\star})^2$, the maximum |
| submit 50 every time | $s^2 + (\bar q - p^{\star})^2 - (0.5 - p^{\star})^2$ |
| be confidently wrong | can be very negative (§1.7) |

Four consequences:

1. **Being at the consensus already earns points.** The brief says "if you match the
   field average on every question, your edge is always zero." That is false for the
   mean *forecast*. Matching the consensus earns $s^2$, the dispersion bonus.
2. **How large is $s^2$?** In the harvested crowd data, the consensus scored about
   $0.222$ and the average individual about $0.252$–$0.256$. So $s^2 \approx 0.03$,
   i.e. $s \approx 0.17$: forecasters typically disagree by about 17 percentage points.
   At 1× that is worth about **+3 RBP per question just for being as good as the
   consensus.** Your realised score was +3.2 per forecast, so most of your points were
   this term (see Part 4).
3. **A sharp, de-vigged bookmaker price is a near-free way to sit at consensus
   quality.** That is why the design is market-first: you don't need to beat the book,
   you need to beat the average punter.
4. **Your submission enters only through $-(q - p^{\star})^2$.** So for expected points,
   relative scoring is still proper: report your best estimate. Being different from
   the crowd is not a goal in itself. Being closer to the truth than the crowd is.

The crowd-bias term $(\bar q - p^{\star})^2$ is the only place where *skill* beyond
calibration earns extra. It is large exactly where the crowd has no sharp line to copy:
obscure counts, timing windows, obscure players. That is where Part 2's modelling effort
goes, and Part 3 is about measuring whether you actually captured it.

### 1.6 Building block: the payoff is asymmetric

Per question, $F_j - B_j$ is at most $F_j$ (when you are perfect, $B_j = 0$). It can be
as low as $F_j - 1$. On an easy question the crowd is right too, $F_j$ is small, and a
correct confident call earns little. A confident wrong call loses almost a full point.

*Worked example (the real worst loss of the round of 32).* "Will DR Congo score a goal?"
was misrouted to a player handler and priced at $q = 0.08$. The crowd was at
$\bar q = 0.37$, and it was YES.

- Your Brier: $(0.08-1)^2 = 0.846$.
- Field: $F \approx (0.37-1)^2 + 0.03 = 0.427$.
- Edge: $\approx -0.42$, i.e. **−84 RBP at 2×**. The record shows −0.45 against the
  consensus and about −90 RBP.

Across the 646 questions with crowd data, the per-question edge against the consensus
has **skewness −1.75**: best +0.38, worst −0.61. **Being wrong while the crowd is right
is the only thing that really hurts.** Almost every safety layer in Part 3 is a response
to this asymmetry.

### 1.7 Building block: integer reports

The API accepts integers 1–99. Rounding to the nearest percent costs at most
$(0.005)^2 = 2.5\times 10^{-5}$ regret, which is negligible. The 1/99 clip costs
$10^{-4}$ when "certain" events happen and saves about $0.02$ when they don't.

### 1.8 Rank versus expected points

The prize is about rank, not expected points. Model your total as
$T \sim \mathcal N(\mu, \sigma^2)$, and suppose you win if $T$ beats a high threshold
$\tau$:

$$
P(\text{win}) \approx \Phi\!\left(\frac{\mu-\tau}{\sigma}\right),
\qquad
\frac{\partial P}{\partial \sigma} = \varphi\!\left(\frac{\mu-\tau}{\sigma}\right)\frac{\tau-\mu}{\sigma^2} > 0 \text{ when } \tau > \mu .
$$

If you are behind ($\tau > \mu$), more variance helps you. This is Lichtendahl & Winkler
(2007): under rank-based payoffs, proper scoring rules stop being proper and you should
*extremize*. The catch is that extremizing a calibrated belief from $p$ to $q$ costs
$(q-p)^2$ of expected Brier per question, and the multipliers scale losses as eagerly
as gains. Part 3 §3.10 shows how you resolved this with simulation. The answer was to
stay honest.

*Calibration of intuition.* The per-question edge standard deviation is about $0.093$.
Over 1,169 questions at 1× that is $0.093\sqrt{1169}\times 100 \approx 318$ RBP of pure
noise in a season total. Luck is large compared with gaps between neighbouring ranks.

### 1.9 Two bots

The rules allowed two bots per account.

- **gigabot1 (Bot A)** is the calibration-first entry. Every price goes down the full
  safety ladder (§3.9). This is the ranked entry.
- **gigabot2 (Bot B)** is the experimental rank-chaser. It takes Bot A's anchored price
  $q$ and pushes it outward by tier, $\sigma(\kappa\,\operatorname{logit}q)$ with caps:
  κ = 3.0 on sharp anchored goal markets and validated-skill markets, 1.8 on props and
  moderate-edge counts, 1.0 on no-edge markets. Near-coin-flip prices
  ($\lvert q - 0.5\rvert < 0.05$) are left alone. This is the sanctioned way to buy
  variance without contaminating Bot A.

---

## Part 2 — Group stage: calibrate to the market, build the machinery

### 2.0 The design principle: one generative core

Every answer about a match comes from one pair of goal intensities
$(\lambda_A, \lambda_B)$. Win, draw, totals, halves, first goal, timing windows, corners
(through their dependence on $\lambda$) and player props (through a tempo scale) are all
functionals of that pair. Two consequences follow:

- **Coherence.** About 15–25 answers per match can't contradict each other. If
  P(win) moves, P(over 2.5) and P(team scores in 2H) move consistently.
- **Information propagates.** When you anchor $(\lambda_A, \lambda_B)$ to the betting
  market, *every* downstream question inherits the market's view, including the many
  no bookmaker quotes.

The pipeline for one fixture:

```
Elo ratings ─► goal model ─► (λA, λB)_model ─┐
                                              ├─► log-blend (w=0.85) ─► (λA, λB) ─► Dixon–Coles grid ─► all goal questions
book odds ─► Shin de-vig ─► multi-book        │                              │
             consensus ─► invert to (λA,λB)_mkt┘                              ├─► Monte-Carlo sim ─► timing questions
                                                                              ├─► count model (μ depends on λ) ─► corners/cards/SoT
                                                                              └─► tempo scale ─► player props (+ book prop lines)
```

### 2.1 Building block: Elo, computed point-in-time

You computed Elo yourself from every international result (martj42's dataset), rather
than scraping it. That makes it reproducible and point-in-time correct by construction.

**Expected result** of A against B, with a 100-point home bonus at non-neutral venues:

$$
W_e = \frac{1}{1+10^{-\Delta R/400}}, \qquad \Delta R = R_A - R_B + 100\cdot\mathbb 1[\text{A at home}] .
$$

**Update** after a result $W \in \lbrace 0, \tfrac12, 1 \rbrace$:

$$
R_A \leftarrow R_A + K\,G\,(W - W_e), \qquad R_B \leftarrow R_B - K\,G\,(W-W_e).
$$

- $K$ depends on the match's importance: 60 for World Cup, 50 for continental finals,
  40 for qualifiers and Nations League, 30 for minor tournaments, 20 for friendlies.
- $G$ is the goal-margin multiplier: 1 for a margin of ≤1, 1.5 for 2, and
  $(11+\text{margin})/8$ for ≥3.

**Why it works: Elo is online logistic regression.** Since
$10^{-x/400} = e^{-x\ln 10/400}$, we have $W_e = \sigma(c\,\Delta R)$ with
$c = \ln 10/400$. The log-loss $-[W\ln W_e + (1-W)\ln(1-W_e)]$ has gradient
$-c(W - W_e)$ with respect to $R_A$. The Elo update is therefore one stochastic-gradient
step on log-loss, with learning rate $K/c$ and $G$ adding scoreline information.

**Point-in-time discipline.** The rating attached to a match is the rating *before*
it, so backtests can't see the future. As a check, the top three (Spain, Argentina,
France) matched eloratings.net.

### 2.2 Building block: from Elo gap to expected goals (Poisson regression)

**Why Poisson for goals.** Split 90 minutes into $n$ small intervals. Suppose each one
contains a goal with probability $\lambda/n$, independently. The count is
$\text{Binomial}(n, \lambda/n)$, and as $n \to \infty$

$$
\binom{n}{x}\left(\frac{\lambda}{n}\right)^x\left(1-\frac{\lambda}{n}\right)^{n-x} \to \frac{\lambda^x e^{-\lambda}}{x!}
$$

because $\binom{n}{x}/n^x \to 1/x!$ and $(1-\lambda/n)^n \to e^{-\lambda}$. Goals are the
classic "many chances, each rarely converted" process.

**The intensity model**, for each side of each match:

$$
\ln \lambda_A = \beta_0 + \beta_1 d + \beta_2 H_A, \qquad \ln \lambda_B = \beta_0 - \beta_1 d,
$$

where $d = (R_A - R_B)/100$ and $H_A = 1$ only for a true home team at a non-neutral
venue. Almost every World Cup match is neutral.

**Estimation: weighted Poisson maximum likelihood.** For observations $i$ (two per
match, one per side) with goals $g_i$, covariates $x_i$ and $\lambda_i = e^{x_i^\top\beta}$:

$$
\ell(\beta) = \sum_i w_i\big[g_i\,x_i^\top\beta - e^{x_i^\top\beta}\big],
\qquad w_i = 0.5^{\,\text{age}_i/730\text{ days}} .
$$

- **Score equation:** $\nabla\ell = \sum_i w_i (g_i - \lambda_i)x_i = 0$. At the
  optimum, weighted residuals are orthogonal to every covariate.
- **Hessian:** $\nabla^2\ell = -\sum_i w_i\lambda_i x_i x_i^\top$, which is negative
  definite. So the log-likelihood is strictly concave and the maximum likelihood
  estimate is unique; BFGS finds it.
- The **half-life** of 730 days was chosen by sweeping 1, 2 and 4 years. The results
  were flat to within ±0.0001 Brier, because Elo already encodes recency. The training
  window is 16 years.
- Weighting by age makes this a **pseudo-likelihood**. Its curvature no longer equals
  the Fisher information, so parameter uncertainty needs the sandwich covariance
  $H^{-1} M H^{-1}$, where $H = \sum w_i \lambda_i x_i x_i^\top$ and
  $M = \sum w_i^2(g_i - \lambda_i)^2 x_i x_i^\top$.

**Interpretation.** You measured $\beta_1 \approx 0.35$. A 100-point Elo edge
multiplies your expected goals by $e^{0.35} \approx 1.42$ and divides the opponent's by
the same factor.

**Why Elo and not per-team attack and defence ratings?** National teams play about 10
matches a year, which is too few to estimate free attack/defence parameters per team.
Elo compresses a century of results into one number, and the *difference* carries the
relative strength.

### 2.3 Building block: the Dixon–Coles scoreline grid

Two independent Poissons under-predict 0-0, 1-1 and other low-scoring draws. Dixon and
Coles multiply the joint pmf by a correction $\tau$ on the four low-score cells:

$$
G[x,y] \propto \tau_{x,y}\,\frac{\lambda_A^x e^{-\lambda_A}}{x!}\frac{\lambda_B^y e^{-\lambda_B}}{y!},
\qquad
\tau_{x,y} = \begin{cases}
1 - \lambda_A\lambda_B\rho & (0,0)\\
1 + \lambda_B\rho & (1,0)\\
1 + \lambda_A\rho & (0,1)\\
1 - \rho & (1,1)\\
1 & \text{otherwise}
\end{cases}
$$

- With $\rho < 0$ (fitted $\approx -0.03$ to $-0.06$), 0-0 and 1-1 gain mass and 1-0
  and 0-1 lose it. That produces more draws, which matches the data.
- $\rho$ is fitted by weighted maximum likelihood on the low-score matches only,
  $\max_\rho \sum_{\text{low}} w_i \ln \tau_{x_i,y_i}$, holding the $\lambda$s fixed.
- The grid is truncated at 12 goals per side and renormalised.
- Intuition: score dynamics create dependence. At 0-0 late on, nobody risks much; at
  1-0 the trailing team commits players forward.

### 2.4 Building block: every goal market is a sum over grid cells

Once you have $G$, the main goal markets are just sums over cells:

| question | formula |
|---|---|
| A wins | $\sum_{x>y} G[x,y]$ |
| draw | $\sum_x G[x,x]$ (the trace) |
| total ≥ $t$ | $\sum_{x+y\ge t} G[x,y]$ |
| both teams score | $\sum_{x\ge1,\,y\ge1} G[x,y]$ |
| A wins by ≥ $n$ | $\sum_{x-y\ge n} G[x,y]$ |
| exactly $n$ margin | $\sum_{\lvert x-y\rvert = n} G[x,y]$ |
| A keeps a clean sheet | $\sum_x G[x,0]$ |
| total is odd | $\sum_{(x+y)\bmod 2=1} G[x,y]$ |

### 2.5 Building block: from odds to probabilities (de-vig)

A bookmaker quotes decimal odds $o_i$. The raw inverses $\pi_i = 1/o_i$ sum to
$S > 1$; the excess $S - 1$ is the margin (overround). The question is which true
probabilities $p_i$ (summing to 1) produced these prices.

**Method 1, basic normalisation:** $p_i = \pi_i / S$. This assumes the margin is spread
proportionally, so it keeps the **favourite–longshot bias** (FLB): longshots are
over-priced in raw odds.

**Method 2, power:** $p_i = \pi_i^{\gamma}$, with $\gamma > 1$ solved so that
$\sum \pi_i^{\gamma} = 1$. Raising to a power above 1 shrinks small numbers relatively
more, so longshots get shaved harder.

**Method 3, Shin (1993)** is the one used live. Suppose a fraction $z$ of bettors are
insiders who know the outcome. A bookmaker defending against them sets prices that
satisfy

$$
\pi_i^2 = S\,p_i\big(z + (1-z)p_i\big).
$$

Solve the quadratic in $p_i$ (with $a = 1-z$, $b = z$, $c = -\pi_i^2/S$) and take the
positive root:

$$
\boxed{\;p_i = \frac{\sqrt{z^2 + 4(1-z)\pi_i^2/S}\; - z}{2(1-z)}\;}
$$

Then solve the one scalar equation $\sum_i p_i(z) = 1$ for $z$ by Brent's method.

- As $z \to 0$ this gives $p_i = \pi_i/\sqrt S$, and summing to 1 forces $S = 1$. With
  no insiders there would be no margin; the size of the margin *reveals* $z$.
- For small $p_i$ the $z p_i$ term dominates, so longshots carry proportionally more
  insurance mark-up, and Shin removes more of it from them. Štrumbelj (2014) found Shin
  beats basic normalisation for soccer.

*Worked example.* Odds 1.80 / 3.60 / 4.75 (home/draw/away) give $\pi$ =
0.5556 / 0.2778 / 0.2105, $S = 1.0439$, a 4.4% margin.

| method | home | draw | away |
|---|---|---|---|
| basic | 0.5322 | 0.2661 | 0.2017 |
| power | 0.5412 | 0.2624 | 0.1964 |
| **Shin** | **0.5387** | **0.2639** | **0.1974** |

The longshot (away) is shaved from 0.2017 to 0.1974 and the favourite gains.

**Player-prop lines** are often one-sided: a book quotes only "Over 0.5" or "Yes". In
that case the code takes $1/o$ and applies an 8% vig haircut, $p \approx (1/o)/1.08$,
instead of silently dropping the line. Dropping it was a real coverage bug.

### 2.6 Building block: many books into one consensus

Several books quote the same match. Treat each de-vigged $p_b$ as a noisy measurement
of the same truth: $p_b = p + \varepsilon_b$ with $\operatorname{Var}(\varepsilon_b) = \sigma_b^2$.
Choose weights $w_b$ that minimise the variance of $\hat p = \sum_b w_b p_b$ subject to
$\sum_b w_b = 1$. With the Lagrangian
$\mathcal L = \sum_b w_b^2\sigma_b^2 - \eta(\sum_b w_b - 1)$, setting
$\partial\mathcal L/\partial w_b = 2w_b\sigma_b^2 - \eta = 0$ gives

$$
w_b \propto \frac{1}{\sigma_b^2} \quad\text{(inverse-variance weighting).}
$$

$\sigma_b$ is never observed, but **the margin is a sharpness proxy**: a 2% book carries
high limits and professional money, an 8% book doesn't. So the code uses

$$
w_b \propto \frac{\text{sharp}_b}{\text{overround}_b + 0.005},
$$

with $\text{sharp}_b = 2.0$ for Pinnacle, Betfair, Kalshi and Polymarket, 1.3 for
bet365, and 1 otherwise. The $0.005$ stops a zero-margin exchange from taking infinite
weight.

*Example.* Pinnacle at a 2.1% margin gets $2.0/0.0258 \approx 77.5$. A soft book at
9.3% gets $1/0.0978 \approx 10.2$. Pinnacle ends up with 88% of the weight. The
consensus is 0.538 / 0.264 / 0.198.

**Why average in probability space here, but in logit space later?** The books are
*correlated noisy copies of one quantity*, since most of them shade the same sharp
line, and the minimum-variance combiner of noisy measurements is an arithmetic mean.
Logit (geometric) pooling is for combining *diverse, independent sources*, such as your
model and the market. A linear pool of independent forecasters is under-confident.

**Prediction markets were checked as a crowd proxy.** On 8 matches, Polymarket sat
within 0.3% of the sharp-book consensus, so deep prediction markets are efficient, not
naive. No market represents the naive contest field, which matters for Part 3.

### 2.7 Building block: inverting the market to intensities

The market gives you probabilities; your machinery runs on intensities. So find the
$(\lambda_A, \lambda_B)$ whose Dixon–Coles grid best reproduces the de-vigged consensus:

$$
(\hat\lambda_A,\hat\lambda_B)^{\text{mkt}} = \arg\min_{\lambda_A,\lambda_B}
\sum_{k\in\lbrace \text{H, D, A, O2.5}\rbrace}\big(\text{pred}_k(\lambda_A,\lambda_B;\rho) - \text{target}_k\big)^2 .
$$

- There are four targets (home, draw, away, over 2.5) and two unknowns, so this is
  **overdetermined least squares**. It is solved in log-space with Nelder–Mead, with
  $\rho$ fixed at the model's value.
- A guard rejects pathological inversions: either $\lambda$ outside $[0.05, 6]$ means
  no anchor.
- Book home/away orientation is matched to the platform's team order before inverting.

*Worked example.* The consensus 0.538 / 0.264 / 0.198, plus P(over 2.5) = 0.52, inverts
to $\lambda^{\text{mkt}} = (1.739, 0.994)$. That grid gives back 0.543 / 0.248 / 0.209 /
0.515. The residual (the draw comes out about 1.6 points low) is model error: a
two-parameter grid can't hit four prices exactly. If an interviewer asks what the
residual means: it measures how far the market's joint view sits from what independent
Poissons with Dixon–Coles can express.

### 2.8 Building block: blending model and market

$$
\boxed{\;\lambda^{\text{blend}} = \exp\!\big(w\ln\lambda^{\text{mkt}} + (1-w)\ln\lambda^{\text{model}}\big) = (\lambda^{\text{mkt}})^{w}(\lambda^{\text{model}})^{1-w},\quad w = 0.85\;}
$$

- **Why log space?** Intensities are positive scale parameters. A log blend treats
  "half the goals" and "twice the goals" symmetrically, for the same reason
  probabilities blend in logit space.
- **Why blend intensities rather than each answer?** Because rebuilding the grid on
  $\lambda^{\text{blend}}$ makes every downstream market inherit the anchor
  *coherently*: halves, first goal, timing windows, and counts whose mean depends on
  $\lambda$.
- **Where 0.85 came from.** It was fitted on 84 real fixtures against FiveThirtyEight's
  published forecasts (as the market-quality anchor). The optimum was ≈0.85 and the
  curve was flat from 0.65 to 1.0. Books replaced 538 live. A comparable independent
  system also settled on 0.85, which raised confidence in it.
- **Motivating failure.** Before anchoring, pure Elo rated the USA at 26% to beat
  Paraguay. The market said about 50%, and the USA won. That one question scored 0.55
  Brier and the match averaged 0.326, worse than a coin flip. Anchoring removed this
  class of blunder.

*Worked example (illustrative coefficients).* A model with $\beta_0 = 0.2$,
$\beta_1 = 0.35$ and an 80-point Elo gap gives $\lambda^{\text{model}} = (1.616, 0.923)$.
Blended with the market's $(1.739, 0.994)$ that becomes
$\lambda^{\text{blend}} = (1.720, 0.983)$, with $\Lambda = 2.703$. That grid gives
**win 0.541, draw 0.250, away 0.209, BTTS 0.520, over 2.5 0.507**. All of §2.10's
examples use this match.

**Anchored goal markets skip three later layers:**

- **Platt calibration:** the book is already calibrated, so re-calibrating would
  double-count.
- **The 25% shrink toward the league-average answer.**
- **The extremizer:** here you *are* the field (§3.1).

### 2.9 Building block: when to submit (closing-line capture)

The closing line is the sharpest price, because it has absorbed lineups and late money.
Scheduler ticks fire in the minutes before each match locks, with a final pass about
two minutes before lock. Lineups drop about an hour out and the book reprices star
props on rotation news, so the closing tick refreshes player-prop lines as well
(`--refresh-props`). Each tick PATCHes only the markets whose price changed.

---

### 2.10 Predicting events the market does not quote

This is where the crowd-bias term $(\bar q - p^{\star})^2$ from §1.5 can be large. You
used a small toolkit of probability results, each applied at the anchored intensities.
Examples use $\lambda^{\text{blend}} = (1.720, 0.983)$ and $\Lambda = 2.703$.

#### A. Poisson thinning and superposition

**Thinning theorem.** Take $N \sim \text{Poisson}(\lambda)$ and keep each event
independently with probability $\theta$. The kept count $N_\theta$ satisfies

$$
P(N_\theta = j) = \sum_{n\ge j}\frac{e^{-\lambda}\lambda^n}{n!}\binom{n}{j}\theta^j(1-\theta)^{n-j}
= \frac{e^{-\lambda}(\theta\lambda)^j}{j!}\sum_{n\ge j}\frac{[(1-\theta)\lambda]^{n-j}}{(n-j)!}
= \frac{e^{-\theta\lambda}(\theta\lambda)^j}{j!}.
$$

So **a thinned Poisson is Poisson with the thinned rate.** Applications:

- **Half questions.** 55.6% of goals fall in the second half (measured on 47k goals),
  so second-half goals for A are $\text{Poisson}(0.556\,\lambda_A)$:
  $P(\text{A scores in 2H}) = 1 - e^{-0.556\cdot 1.720} = 0.616$.
- **Half-time states.** A Dixon–Coles grid at $(h_1\lambda_A, h_1\lambda_B)$ gives
  P(level at half-time) = 0.420, plus "goalless at HT", "A leads at HT", and so on.
- **Goal flavours.** Headers are 21% of open-play goals, so
  $P(\text{headed goal}) = 1 - e^{-0.21\Lambda}$. This beat a flat rate by +0.027 Brier.
  Outside-the-box goals are 13% and beat flat by +0.021.
- **Stoppage time.** WC2022 shares were 3.2% of goals in first-half added time and
  8.9% in second-half added time, so $P = 1 - e^{-\Lambda \cdot \text{share}}$.

**Superposition** is the reverse: independent Poisson processes add. Total goals are
$\text{Poisson}(\Lambda)$.

#### B. First-goal questions (competing exponentials)

Merge A's and B's goal processes. Given that a goal occurs, it is A's with probability
$\lambda_A/\Lambda$. That holds even when the hazard varies over the match, provided the
time profile is shared. So

$$
P(\text{A scores first}) = \frac{\lambda_A}{\Lambda}\big(1 - e^{-\Lambda}\big) = 0.594 .
$$

- **First goal of the match comes in the 2H** means no goal in H1 and at least one in
  H2: $e^{-h_1\Lambda}(1 - e^{-h_2\Lambda}) = 0.234$.
- **Scores first and wins.** Multiply P(first) by a conditional win probability equal
  to the side's win probability shifted by +1.34 logits. That shift reproduces the
  measured fact that 75.7% of first scorers win (15,381 internationals).

#### C. Parity and exact-count questions (where the crowd uses heuristics)

**Total goals odd.** For $N \sim \text{Poisson}(\Lambda)$,
$\mathbb E[(-1)^N] = e^{-\Lambda}\sum_n \frac{(-\Lambda)^n}{n!} = e^{-2\Lambda}$, so
$P(\text{even}) - P(\text{odd}) = e^{-2\Lambda}$ and

$$
P(\text{odd}) = \frac{1 - e^{-2\Lambda}}{2} < \frac12 .
$$

At $\Lambda = 2.703$ that is 0.498, and the Dixon–Coles grid (which adds mass at 0-0 and
1-1, both even) brings it down to **0.486**. The crowd anchors at 50.

**Other heuristic traps:**

- **Margin exactly 1** is 0.373 on the grid. The crowd tends to conflate "one-goal game"
  with "not a draw" and prices it near 73.
- **Either team wins both halves** is 0.208 (halves treated as independent grids at
  $h_1\lambda$ and $h_2\lambda$; the Monte-Carlo engine confirms the cross-half
  dependence is negligible). The crowd reasons "the better team wins both" and prices
  it near 35.
- **Halves have equal goals.** Given $N = 2k$ total goals, each lands in H1 with
  probability $h_1$, so $P(\text{equal}) = \sum_k P(N=2k)\binom{2k}{k}h_1^k h_2^k$.
  This binomial split was validated on 11,658 matches.

#### D. Timing windows: the minute profile

**Inhomogeneous Poisson process.** Events arrive with a minute-varying hazard. You
fitted a 15-minute bucket profile $\alpha_b$ for each family (goals, corners, cards,
fouls, shots on target, offsides). A window $[\ell, u)$ holds a share of the total
intensity equal to

$$
\phi_{[\ell,u)} = \frac{\sum_{m=\ell}^{u-1} e^{\alpha_{b(m)}}}{\sum_{m=0}^{89} e^{\alpha_{b(m)}}},
\qquad
P(\ge 1 \text{ goal in window}) = 1 - e^{-\Lambda\,\phi} .
$$

The hydration breaks fall at the 22nd minute of each half, i.e. match minutes 22 and
67. (An earlier guess of [75, 90) for the post-break window under-predicted.)

| window | intensity share | P(goal) at $\Lambda$ = 2.703 |
|---|---|---|
| before break 1, [0, 22) | 0.173 | 0.374 |
| rest of H1, [22, 45) | 0.240 | 0.477 |
| between breaks, [22, 67) | 0.501 | 0.742 |
| after break 2, [67, 90) | 0.326 | 0.586 |

**The insight: the hazard rises through each half and through the match.** A flat-time
assumption would put [0, 22) at 22/90 = 0.244 and P = 0.483, about 11 points too high.
For count windows (corners and offsides before break 1, cards after break 2), the same
share scales the count model's mean, and the negative-binomial tail gives the
probability.

**How the profile is estimated, and a trap.** Model the hazard in minute slot $m$ for a
team as

$$
h(m) = c_{\text{match,team}}\cdot e^{\alpha_{b(m)} + \beta_{\text{state}(m)}},
$$

where the state is the team's score margin, clipped to ±2.

- **Non-goal families: the conditional-multinomial trick.** Given a (match, team)
  total of $n$ events, the slot counts are multinomial with cell probabilities
  $h(m)/\sum h$. The nuisance $c_{\text{match,team}}$ cancels, leaving the conditional
  log-likelihood
  $\sum_{\text{events}}[\alpha + \beta] - \sum_{\text{groups}} n\ln\sum_m e^{\alpha+\beta}$.
  This is valid because corners and cards don't change the score, so the state path is
  exogenous to them.
- **Goals: the trick is biased.** Each goal *creates* the state that follows it, so the
  total is not ancillary for the shape. Your synthetic-recovery test caught this: the
  fitted late bucket came out at +1.20 when the truth was +0.50. The fix is the full
  point-process likelihood $\sum_m[k_m\ln h(m) - h(m)]$ on the realised path, with
  $\ln\lambda$ from the goal model as an offset. That recovered every parameter to
  ±0.08. **Lesson for interviews: "condition on the total to kill nuisances" silently
  fails for self-exciting processes, and you should always test estimators on data
  simulated from known parameters.**

#### E. The Monte-Carlo engine (timing questions)

For questions where state dependence matters (level at HT, leading at HT, more goals in
H2 than H1, which side scores the first goal of a half, scores first and wins),
`mc_engine.py` simulates 50,000 matches on 90 one-minute slots:

- In each slot, each team scores $\text{Poisson}(\lambda\cdot e^{\alpha_{b} + \beta_{\text{state}}}/Z)$
  goals, with the state updated after every goal. Trailing teams score more: the fitted
  multipliers are about ×1.28 one goal down and ×1.33 two or more down. $Z$ is
  calibrated so the simulated mean total equals $\lambda$.
- **Only goals are state-coupled.** The ablation showed count-family coupling *hurts*
  out of sample (shots on target −0.018 Brier). In-sample multipliers of that size
  don't transfer across tournaments, which is the same lesson the referee effects
  taught.
- The simulator *generates* the low-score dependence that Dixon–Coles patches by hand,
  and it removes a roughly 2-point over-price that the closed-form independent-halves
  handler carries.
- Questions are answered as the frequency of the event across the draws.

#### F. Count questions: the negative binomial

**Overdispersion.** Corners per team per half have variance/mean ≈ 1.42, while a
Poisson forces 1. Introduce a match-level "type of game" multiplier
$G \sim \text{Gamma}(k, 1/k)$, with $\mathbb E G = 1$ and $\operatorname{Var} G = 1/k$,
and let $X \mid G \sim \text{Poisson}(\mu G)$. Integrating out $G$:

$$
P(X=x) = \int_0^\infty \frac{e^{-\mu g}(\mu g)^x}{x!}\frac{k^k g^{k-1}e^{-kg}}{\Gamma(k)}\,dg
= \frac{\Gamma(x+k)}{x!\,\Gamma(k)}\left(\frac{k}{k+\mu}\right)^k\left(\frac{\mu}{k+\mu}\right)^x .
$$

This is the negative binomial. By the law of total variance,
$\operatorname{Var}X = \mathbb E[\mu G] + \operatorname{Var}(\mu G) = \mu + \mu^2/k$, so
$k$ follows by method of moments:

$$
k = \frac{\mu^2}{\operatorname{Var} - \mu}\quad(\text{Poisson if } \operatorname{Var} \le \mu).
$$

**Mean model.** For family $f$ and half $h$:

$$
\ln\mu = c_0[f,h] + c_1[f]\ln\lambda_{\text{team}} + c_2[f]\ln\lambda_{\text{opp}} .
$$

Here $\lambda$ stands in for attacking volume (the corners–shots correlation is 0.60,
while corners–goals is only 0.11). Cards use only an intercept and a half effect,
because $\lambda$ covariates hurt them in backtest. The point is that **counts inherit
the market anchor**: a book that makes the match one-sided implies more corners for the
favourite, without anyone quoting a corners line.

**Comparison questions** ("Will A have more corners than B?"):

$$
P(A > B) = \sum_{a\ge 1} f_A(a)\,F_B(a-1),
$$

where $f_A$ is A's pmf and $F_B$ is B's cdf. **A tie settles as NO.** With equal means
of 4.4 and $k = 5.3$, $P(A>B) = 0.446$ and $P(\text{tie}) = 0.109$, so *both* teams'
"more corners" probabilities sit well below 0.5. This is a trap for intuition that the
model handles automatically.

**Sums and thresholds:** convolve the two pmfs, $f_{A+B} = f_A * f_B$, then
$P(\text{sum}\ge t) = \sum_{s\ge t} f_{A+B}(s)$. Full-match counts convolve the two
halves.

**Joint questions with dependence.** "More corners AND more shots" can't use
independence, because a team having an attacking day gets both. The code adds a
linearised copula correction:

$$
P(\text{both}) \approx p_1p_2 + \rho\sqrt{p_1(1-p_1)p_2(1-p_2)} .
$$

The measured residual indicator correlation, after removing team quality, was 0.41 on
314 matches. You shipped $\rho = 0.30$, deliberately below it. An earlier value of 0.05
was **six times too low** and made this the one handler positioned to lose to the
crowd. "Each team records ≥ $N$ shots on target" uses $\rho = -0.05$ (the raw −0.23
correlation is mostly matchup mix that the anchored marginals already encode).

**Other count-derived questions:**

- **Concentration curves.** P(some player takes ≥2 of the team's goals | team scored
  $N$) = 0, 0, 0.156, 0.42, 0.77, ... for $N$ = 0, 1, 2, 3, 4, ...; P(any brace) sums
  this curve against the team goal distribution.
- **Goalkeeper saves.** Saves = shots on target faced − goals conceded, an identity that
  held in all 628 StatsBomb matches. Sum over the shots-on-target pmf, thinning each
  shot with save probability $1 - \lambda_{\text{opp}}/\mu_{\text{SoT}}$.
- **Distinct shooters.** P(≥5 different players shoot | $N$ team shots) is a logistic
  in $N$, with coefficients −5.333 and 0.847, fitted on 628 team-matches. Shot volume
  is a sufficient statistic.

#### G. Player props

**Step 1: a per-minute rate with shrinkage (Gamma–Poisson conjugacy).** Put a prior on
the rate, $r \sim \text{Gamma}(a, b)$ with density $\propto r^{a-1}e^{-br}$, and observe
$y$ events in $m$ minutes with likelihood $\propto r^y e^{-rm}$. The posterior is
$\text{Gamma}(a+y, b+m)$. Choosing $a = r_0 M_0$ and $b = M_0$ (that is, "$M_0$
pseudo-minutes at the prior rate $r_0$") gives the posterior mean

$$
\hat r = \frac{y + r_0 M_0}{m + M_0},
$$

with $M_0 = 270$ minutes for shots on target and 540 for goals and goal-or-assist. Rarer
events need heavier shrinkage, which the backtest confirmed rather than assumed.

**Step 2: centre the prior on the player, not the population.** This was a real leak.
Shrinking 270–540 minutes toward "average World Cup player" pulls exactly the elite
outliers the questions ask about. Player prices sat 18–28 points under the crowd, and
the gap grew with the crowd's level, which is the signature of over-shrunk stars. The
fix:

$$
r_0 = \tau \cdot r_{\text{club}}, \qquad \tau = 0.80 .
$$

- $\tau$ is a club-to-country discount (tougher opponents, less familiar team-mates).
  Its optimum was flat on settled player questions.
- Club rates came from Understat (top leagues: xG, shots, xA) and API-Football (broad
  leagues, with shots on target directly). Together they cut the **coverage hole**, the
  share of prop players with no rate, from 44% to 16%. Where Understat lacks shots on
  target, it is estimated as shots × 0.37.

*Example.* A club rate of 0.45 shots on target per 90 gives $r_0$ = 0.36 per 90, i.e.
0.004 per minute. After one 90-minute match with 2 shots on target:

$$
\hat r = \frac{2 + 0.004\cdot 270}{90+270}\times 90 = 0.77 \text{ per 90}.
$$

**Step 3: tempo and scope.** Multiply the rate by $\Lambda/\Lambda_{\text{league}}$, so
the same player is priced higher in an open tie, and by $h_1$ or $h_2$ for half-scoped
questions.

**Step 4: the minutes mixture (Jensen).** $f(M) = 1 - e^{-rM}$ has
$f''(M) = -r^2 e^{-rM} < 0$, so it is concave, and by Jensen
$\mathbb E[f(M)] \le f(\mathbb E M)$. **Plugging in expected minutes overstates the
probability.** The correct computation mixes over scenarios:

$$
P(\ge k) = \sum_{s} w_s\Big[1 - F_{\text{Pois}}\big(k-1;\; r\cdot\text{tempo}\cdot M_s\big)\Big],
\quad (w_s, M_s) \in \lbrace (0.80, 80'), (0.12, 25'), (0.08, 0') \rbrace .
$$

*Example:* with $r = 0.02$ per minute, the mixture gives 0.686 against 0.738 from
plugging in $\mathbb E M = 67$, a 5-point overstatement. The bias is largest exactly for
rotation risks.

**Step 5: anchoring to book prop lines (§3.5) and converting thresholds.** Books quote
only the "1+ shots on target" line, $p_1 = P(\ge 1)$. Treating shots on target as
Poisson($\mu$), invert to $\mu = -\ln(1-p_1)$ and read off
$P(\ge k) = 1 - F_{\text{Pois}}(k-1;\mu)$. This keeps the book's ranking of players.
Mbappé's $p_1 = 0.834$ gives $P(\ge 2) = 0.536$, and Isak's $0.478$ gives $0.139$.
Before this fix, Mbappé's 2+ line was priced at 19 against a crowd at 63, and it landed.

**Step 6: unresolved players.** If neither the club table nor tournament minutes match
a name, the model would price the naked population floor. That error class once had
Haaland at 8% against a book at 52%. Instead, blend 50/50 with the crowd's typical level
for that question type.

**Name resolution is a statistical issue too.** "Kylian Mbappé (France, #10)" matched
no row, so the best striker on the pitch collapsed to the prior. Ødegaard and Sørloth
lost their book anchors because Nordic letters were dropped rather than transliterated.
The fuzzy join requires the last name to match and the first names to align, and it
returns *nothing* on ambiguity, so the code falls back rather than guesses.

#### H. Knockout specials: advancing, extra time, shootouts

$$
P(\text{advance}) = P(\text{win}_{90}) + P(\text{draw}_{90})\Big[P(\text{win}_{ET}) + P(\text{draw}_{ET})\,s_{\text{pen}}\Big]
$$

- **Extra time** is a Dixon–Coles grid at $(0.35/3)\cdot\lambda$: 30 minutes, scoring
  at 0.35× the pro-rated rate, grounded on 839 extra-time matches. 81% of those still
  reached penalties.
- **The shootout** is near a coin flip with a mild favourite tilt,
  $s_{\text{pen}} = 0.5 + 0.2(r - 0.5)$, where $r = P(\text{win}_{90})/(P(\text{win}_{90}) + P(\text{lose}_{90}))$.
  Team evidence overrode the tilt for the final (Argentina 0.55, Spain 0.45).
- **Validation:** the decomposition landed within 1–2 points of the book's outright
  price on every leg.
- **Coherence:** P(A advances) + P(B advances) = 1.
- Extra time happening equals $P(\text{draw}_{90})$, and a shootout equals
  $P(\text{draw}_{90})P(\text{draw}_{ET})$.
- A recurring parser trap: "win the tie" must not be read as "win in 90 minutes".

#### I. Base-rate events

| question | model |
|---|---|
| penalty awarded | 0.293 (6 tournaments). WC26 ran at about 60% of that (λ = 0.206/match), used for "penalty scored" via $1 - e^{-\lambda \cdot \text{conversion}}$ with hard bounds [0.10, 0.30] |
| red card | 0.065 |
| penalty or red | $1-(1-0.293)(1-0.065) = 0.339$ |
| own goal | flat 0.10 (scaling with total goals was tested and lost) |
| a substitute scores | rate $\theta = -\ln(1-0.18)/\Lambda_{\text{league}}$, then $1 - e^{-\Lambda\theta}$, so it scales with the match's goals |
| ≥ $N$ substitutions | a mixture of empirical exceedance rates, weighted by P(draw after 90), because extra-time-bound teams hold subs back |
| VAR on-field review | Poisson, $\lambda = 0.33\,(1 + P(\text{draw})/3)$ |
| first card before the first goal | competing risks with time-varying hazards: $\sum_m h_c(m)S(m)$, where $S$ is joint survival; a 0.6/0.4 mix of two settlement readings |
| first scorer wears 1–9 | P(a goal ever) × each team's single-digit goal share, weighted by intensity, with own goals folded in |

#### J. The calibration layer (Platt, per family)

**Why calibration is fixable after the fact: the Murphy decomposition.** Bin your
forecasts into groups $g$, each with $n_g$ forecasts, mean forecast $\bar q_g$ and
realised frequency $\bar y_g$; let $\bar y$ be the overall base rate. Then

$$
\bar B = \underbrace{\tfrac1N\textstyle\sum_g n_g(\bar q_g - \bar y_g)^2}_{\text{reliability (want 0)}} - \underbrace{\tfrac1N\textstyle\sum_g n_g(\bar y_g - \bar y)^2}_{\text{resolution (want large)}} + \underbrace{\bar y(1-\bar y)}_{\text{uncertainty}} .
$$

Post-processing can fix reliability but can't create resolution.

**Platt scaling:** $q' = \sigma(a\cdot\operatorname{logit} q + b)$, fitted by logistic
maximum likelihood. $a < 1$ compresses over-confidence and $b$ shifts bias. Values in
production:

| family | $(a, b)$ | note |
|---|---|---|
| goal | (0.842, 0.003) | only for **unanchored** goal questions |
| count | (0.972, 0.181) | always applied |
| player | (1, 0) | 5-fold cross-validation on live settled props: neutral 0.2058 beat the old fit 0.2065 and a re-fit 0.2092. The re-fit over-fitted, with fold slopes from 0.29 to 0.69 |
| offsides | (1, 0.031), intercept only | its own family, because of a definition gap between StatsBomb and the platform: live predictions averaged 0.50 against 0.125 realised. A ridge penalty $b^2/(2\cdot0.8^2)$ keeps the shift small |

When forecasts barely vary (every "2+ cards in 2H" sits near 0.63), the slope $a$ can't
be identified, so the code falls back to intercept-only.

#### K. Data integrity is the largest error source

The most expensive mistakes weren't modelling errors. They were questions routed to the
wrong handler, which then priced confidently wrong:

- "Both teams have at least one shot on target" was swallowed by a greedy player regex
  and priced at about 9% against a fair 75%.
- "Will DR Congo score?" was routed as a player and priced at 8% against roughly 40%
  (§1.6).
- "At halftime, will ..." lost its half scope and priced as full-match, about 0.88
  against 0.60.

The structural response was to **ledger every unparsed question loudly** (the drop
audit went 56 → 12 → 0), and to build new knockout templates from validated primitives.
A backtest of the 9 recovered archetypes on 58 settled matches (702 questions) showed
**+0.0237 pooled Brier skill against climatology** when anchored, so the recovery was
real signal, not just tidying.

---

## Part 3 — Testing the edge, then spending it where it exists

### 3.0 The problem in one sentence

You are betting into a counterparty, the crowd, **whose price you can't see before
lock**, and you only earn beyond the dispersion bonus where you differ from it *and*
are right.

### 3.1 Where edge can come from, from §1.5

Recall
$\mathbb E[\text{RBP}]/(100m) = s^2 + (\bar q - p^{\star})^2 - (q - p^{\star})^2$.

- **Book-priced main markets (win, totals, BTTS).** The crowd reads the same odds you
  anchor to, so $\bar q \approx$ book $\approx q$, and you earn about $s^2$. The anchor
  is a *defensive floor* there, not a weapon. Your closing-line value is zero by
  construction, because you *are* the close.
- **No-line markets** (counts, timing windows, obscure props). The crowd clusters near
  naive base rates or round numbers, so $(\bar q - p^{\star})^2$ can be large, and a
  model that tracks $p^{\star}$ captures it. **Edge lives here.** But any modelling
  error is also uncapped by a book, so this is also where the blow-ups live.

### 3.2 Step 1 (before any data): a structural field model plus an edge-aware extremizer

**Field model.** For each family, assume the crowd captures a fraction $\lambda_f$ of
your log-odds signal:

$$
f = \sigma\big(\lambda_f\cdot\operatorname{logit}p\big),
\qquad \lambda_f^{\text{prior}}:\ \text{goal } 0.85,\ \text{player } 0.45,\ \text{count } 0.40,\ \text{timing } 0.35 .
$$

**Extremize the deviation from the field, not the probability itself:**

$$
\operatorname{logit}q = \operatorname{logit}f + \kappa\big(\operatorname{logit}p - \operatorname{logit}f\big), \qquad \kappa \ge 1 .
$$

- If $p = f$, then $q = f$. There is no gamble where you have no disagreement.
- If $p \ne f$, the disagreement is amplified by $\kappa$, so variance is concentrated
  exactly on your edge.

**A neat composition.** Substituting the field model gives

$$
q = \sigma\Big(\big[\lambda_f + \kappa(1-\lambda_f)\big]\operatorname{logit}p\Big).
$$

A clueless field ($\lambda_f \to 0$) gives full extremizing (coefficient $\kappa$). A
sharp field ($\lambda_f \to 1$) gives none (coefficient 1). The one number $\lambda_f$
*is* your assumed edge.

*Example (timing family, $\lambda_f = 0.35$, $\kappa = 1.3$).* A belief of 0.30 implies
a field of 0.426 and a submission of 0.267. A belief of 0.60 implies 0.535 and 0.619.

**Two audits shaped this:**

- **Phase A:** the public API exposed no field data at all, only your own predictions.
- **Phase B:** Polymarket matched sharp books to 0.3%, so no market proxies a naive
  crowd. The field model had to be structural.

### 3.3 Step 2: the live edge gate (first settled matches)

**Recover outcomes from your own Brier.** The results endpoint returns $B$ and your
$q$. Then $y = 1$ if $(q-1)^2 = B$ and $y = 0$ if $q^2 = B$. This is ambiguous at
$q = 0.50$, where both give 0.25; those rows are dropped and documented.

**The gate.** For each family (count, player, timing):

$$
\kappa_{\text{family}} =
\begin{cases}
1.3 & \text{if } n \ge 20 \text{ and } \bar B_{\text{us}} < \bar B_{\text{field proxy}} - 0.005\\
1 & \text{otherwise}
\end{cases}
$$

$\kappa$ was meant to rise by stage (×1.5 at 2×, ×2 at 3×), but the live call doesn't
pass the stage, so in practice it stayed at 1.3.

**Result:** **timing confirmed** ($\kappa = 1.3$) on the group-stage settlements,
committed at the start of the round of 32; count and player not confirmed.

**Weak point, and how to answer it.** The "field proxy" is your own belief shrunk
toward 50. So the test asks whether your sharpness pays against a duller copy of
yourself, which really means whether resolution beats miscalibration. That is
*necessary but not sufficient* for beating the real crowd. You knew this, which is why
§3.6 (harvesting the real crowd) mattered so much. The real data later agreed on
timing: +15 RBP against the *consensus* on 21 questions.

### 3.4 Step 3: an offline backtest gate (six past tournaments)

Waiting for 20 live settlements per family risks the gate never firing. So you measured
per-family edge on past tournaments (the goal backtests cover WC 2018, Euro 2020,
WC 2022, AFCON 2023, Copa 2024 and Euro 2024; the StatsBomb count backtests cover the
last five) against a deliberately *sharp* field proxy ($\lambda = 0.70$), using a
**paired t-test** on per-question Brier differences:

$$
D_j = B^{\text{field}}_j - B^{\text{us}}_j, \qquad t = \frac{\bar D}{\hat\sigma_D/\sqrt n} .
$$

- **Player props:** 0.1147 against 0.1222, $t = 18.7$, $n = 12{,}120$. That is
  overwhelmingly significant.
- **But you over-rode it** with a **survivorship guard**. Backtest players were in the
  data *because* they had international history, while live World Cup squads were full
  of players without it (Kubo: 90 career minutes, priced at 7% to score or assist, did
  it, Brier 0.865). The gate stayed at κ = 1.
- Count and timing were not significant.

**The interview lesson:** a significant test on the wrong population is not edge.
Ask what generated the sample.

### 3.5 Step 4: the closing-line value ledger

Every run priced every market three ways and logged all three:

$$
q^{\text{pure}}\ (w = 0), \qquad q^{\text{blend}}\ (w = 0.85,\ \text{submitted}), \qquad q^{\text{close}}\ (w = 1) .
$$

After settlement:

- **Anchor value** $= \bar B^{\text{pure}} - \bar B^{\text{close}}$: did the book beat
  the model?
- **CLV of the blend** $= \bar B^{\text{close}} - \bar B^{\text{blend}}$: did your 15%
  model share beat the close? If not, it's noise and should be cut.

**What it decided: the player-prop book weight.** On 51 settled book-anchored props,
Brier was **monotone toward the book**: pure model 0.250, blend at 0.85 0.206, pure book
0.202. The model is data-starved on exactly the stars the book prices well. The fix
interpolates off $w$:

$$
w_{\text{player}} = 1 - 0.20\,(1 - w) = 0.97 .
$$

*Why even 15% hurt: the logit drag.* Raúl Jiménez's model was at 7%
($\operatorname{logit} = -2.59$) and his book line at 27% ($-0.99$). A 0.85 blend gives
0.226, 4.4 points below the book. At 0.97 the blend gives 0.261. He scored. A
confident-low model has a large negative logit and pulls hard even at small weight, and
the 2× multiplier doubles the damage.

### 3.6 Step 5: harvesting the real crowd (the turning point, about Jun 19)

**The discovery.** The public API hid the crowd, but the logged-in web app's
`match-stats` endpoint returned, for each **settled** question, the crowd mean
$\bar q_j$, the field Brier $F_j$, your Brier, and the per-question relative score.

**The hard constraint.** The crowd only appears *after* a market locks. **It can never
be a live input**, only a calibration target and grader. This was enforced in the
design from then on.

**The dataset** (`settled_with_crowd.parquet`) joins, for each question, your $q$, the
recovered $y$, the authoritative archetype label, and $\bar q$. The headline metric was
**edge against the consensus**:

$$
e_j = (\bar q_j - y_j)^2 - (q_j - y_j)^2 .
$$

This is a *tougher* benchmark than the scoring baseline, since by §1.4
$F_j = (\bar q_j - y_j)^2 + s_j^2$. Beating the consensus means beating the field by
more than $s^2$.

- **Reflexivity check.** Your own forecast is inside $\bar q$. The leave-one-out
  consensus $(N\bar q - q)/(N-1)$ differs by $O(1/N)$, and with $N \approx 2{,}800$ that
  is under 0.0004, so it was ignored and documented.
- **First reading** (257 questions, 26 matches): consensus 0.224, you 0.232, average
  individual 0.252. You were ranked 240 of 2,818 at +520 RBP. **You beat the scatter
  but lagged the consensus.**
- **Labels before conclusions.** Six rows of an already-fixed parser bug were labelled
  `player_sot_ge`. Excluding them flipped that archetype from −83.6 to +11.7: a strength,
  not a leak.

### 3.7 What the crowd data said

**(a) The deviation-size risk curve.** These are the 646 settled questions with crowd
data, recomputed for this guide:

| $\lvert q - \bar q\rvert$ (points) | n | mean edge per question | share of questions won |
|---|---|---|---|
| 0–5 | 333 | +0.0013 | 0.48 |
| 5–15 | 254 | +0.0007 | 0.59 |
| 15–25 | 37 | −0.0058 | 0.65 |
| **> 25** | **22** | **−0.079** | 0.55 |

**Moderate disagreement paid. Large disagreement (25+ points) was strongly negative, and
historically every one of those losses was a bug-class extreme.** This curve is the
justification for the crowd clamp (§3.9i).

**(b) Over-confidence, by confidence bucket** (an earlier 253-question cut):

| lean $\lvert q - 0.5\rvert$ | n | hit rate | RBP per question |
|---|---|---|---|
| 0.0–0.1 | 88 | 55% | +0.09 |
| **0.1–0.2** | 84 | 63% | **+0.67** |
| 0.2–0.35 | 56 | 66% | −0.15 |
| **0.35–0.5** | 25 | **76%** | **−1.20** |

The most confident bucket hit 76% of the time and still lost, because each miss costs
about 27 points and each hit saves only a few. **Points live in moderate, well-aimed
leans.**

**(c) The edge is bimodal by question type.** There is real edge in team corners and
shots-on-target counts, timing, and `player_scores`. There are bleeds in offsides,
fouls and cards.

**(d) Fouls and offsides are anti-signal.** A *flat* value beat both you and the crowd:

- fouls: flat 0.245, crowd 0.263, you 0.277 (n = 21);
- offsides: flat at 0.40 scored 0.233, crowd 0.253, you 0.280 (n = 22).

Your match-to-match variation there carried no information. By §1.2, uncorrelated
variation adds pure $(q - p^{\star})^2$.

**(e) Player props: the crowd beat the book.** On 29 settled anchored props the crowd
scored 0.223 against the book's 0.258. The root cause is that the book is
miscalibrated on the low side: book prices under 0.34 realised 0.44.

**(f) Power analysis: why almost nothing per question type was identified.** The
per-question edge has $\hat\sigma \approx 0.093$. To detect a mean edge $\delta$ at
$t = 2$ you need $n \approx (2\hat\sigma/\delta)^2$:

| edge to detect $\delta$ | questions needed |
|---|---|
| 0.005 | ≈ 1,380 |
| 0.01 | ≈ 345 |
| 0.02 | ≈ 86 |

Per-archetype samples were 5–50. **Almost no archetype-level effect was statistically
identified.** That single fact explains the style of every adjustment that follows:
structural rather than fitted, shrunk, cross-validated by match, and deliberately
under-shifted.

### 3.8 The overfitting controls (the meta-method)

These are worth stating crisply in an interview:

1. **Leave-one-MATCH-out cross-validation.** Questions within a match are correlated
   through the shared $\lambda$, so the CV unit is the match, not the question. A single
   outlier match can't be memorised.
2. **A parsimony rule.** Choose the *smallest* parameter within $\varepsilon$ ($10^{-3}$)
   of the best out-of-sample Brier, rather than the argmin.
3. **A no-regression gate.** Ship a change only if cross-validation improves by at
   least $5\times10^{-4}$. Otherwise keep the identity.
4. **Ridge toward identity.** For example, the confidence fit minimises
   $\overline{B} + 8(a-1)^2 + 1.5\,b^2$.
5. **Pre-registration.** The de-extremize hypothesis was scored with $\gamma = 0.78$
   fixed in advance, from an append-only ledger of fresh settlements, so it couldn't be
   tuned on the test set.
6. **Pseudo-observation shrinkage.** Per-type estimates shrink toward the pooled value:
   $\hat\theta_a = (n_a\bar\theta_a + 5\bar\theta_{\text{pool}})/(n_a + 5)$.
7. **Deliberate under-shift** when the fitted optimum sits on the edge of the search
   grid, i.e. is unidentified (§3.9g).
8. **Decision rules fixed before experiments.** A squad-quality feature and a
   gradient-boosted hybrid were both rejected by the out-of-sample rule over six
   tournaments. With one parameter and 48 observations, the prior does the inference.

### 3.9 Spending the edge, one question type at a time

This is the full Bot A ladder in `runner.py`. Each rung moves the price only if it has
something measured to add. **Order matters: confirmed team news goes last because it
beats everything above it.**

1. Model price from the handlers.
2. Platt calibration by family. Skipped for anchored goal markets.
3. Book anchor. Player props are anchored hardest, at about 0.97.
4. Shrink 25% toward the league-average answer (unanchored questions only).
5. Extremize, where a family has earned it (timing, κ = 1.3).
6. Confidence floor on confident-low player bets (unanchored only).
7. LLM panel (no book line, gated, mostly shadow).
8. Risk gate: shrink no-edge archetypes toward their base rate.
9. Crowd hug on flat event kinds.
10. Endgame variance dial (off by default).
11. Lineup caps for confirmed benched or out players.

After the ladder come the submit-time safety layers: the crowd clamp, hand-written
overrides, lineup-conditional rules, the anchor-regression guard, and audits.

#### (a) Book-priced main markets: win, draw, totals, BTTS

Pure anchor. No Platt, no extremizing, LLM excluded. Expected edge is about $s^2$. Bot B
pushes these hardest, since that is where you are *directionally* right against a noisy
field.

#### (b) Grid-derived no-line goal markets

First goal, margins, win both halves, odd total, goalless at HT, timing windows. These
are where the crowd uses provably naive heuristics (§2.10C/D), and they inherit the
anchor. Validated at +0.0237 skill on recovered markets.

#### (c) Timing (MC engine plus κ = 1.3)

The one family with confirmed live edge from the group-stage settlements. Extremized around the
structural field (§3.2). LLM shadow-only, which protects proven edge.

#### (d) Team counts: corners and shots on target

Real model edge. The LLM is shadow-only here as well. These are on the endgame
allow-list.

#### (e) Fouls and offsides: the risk gate

Shrink toward a pinned base rate in logit space:

$$
\operatorname{logit}q = \operatorname{logit}b + s\big(\operatorname{logit}p - \operatorname{logit}b\big), \qquad s = 0.15 .
$$

- **Fouls base 0.50 by a symmetry argument.** "A commits more fouls than B" compares
  exchangeable teams, so $P(A>B) = P(B>A) \le 0.5$ once tie mass is removed. The
  realised rate jumped around between samples (0.57 at n = 21, 0.44 at n = 16), so the
  symmetric value is the robust one. A model at 0.62 ships 0.518.
- **Offsides base 0.40.** The crowd's 0.49 shrunk toward the realised 0.36: the
  definition gap makes even the crowd too high. A model at 0.20 ships 0.365.
- Cards use $s = 0.5$, and both-teams shots on target uses $s = 0.3$ (base 0.70).
- **The two layers used to fight each other.** Risk-gated types used to be extremized
  too: one layer inflated the noise the other deflated. Excluding them from the
  extremizer fixed it. This kind of bug never throws an exception.

#### (f) Cards

Card timing types are excluded from the count Platt, which shifts prices up while cards
ran low all through WC26. The late-tournament crowd file confirmed it: on the 30 most
recent total-cards questions the realised rate was 0.30 while the shipped average was
0.51. The raw structural model at about 0.28 was right.

#### (g) Player props, in five layers

1. **Book anchor at 0.97** (§3.5).
2. **Crowd lift.** You can't see the crowd before lock, but the crowd-minus-book premium
   per prop type was stable enough to predict. It is estimated on settled anchored props
   and shrunk toward the pooled value with 5 pseudo-observations: scorer +0.041, shots on
   target +0.074, goal-or-assist +0.003. The lift is
   $$q \leftarrow q + \min(0.5\cdot\text{premium},\ 0.06), \quad \text{only if } q < 0.85 \text{ and only on the live pass.}$$
   **The magnitude was not identified.** The leave-one-out Brier-against-shift curve was
   monotone with no interior optimum, the raw edge was borderline ($p \approx 0.065$),
   and the direction was a 15–14 coin flip. So the factor was cut from 0.75 to 0.5,
   which keeps about 70% of the modelled gain for roughly half the downside, and the
   6-point cap does the real regularising. *Example:* a model at 0.10 and a book at
   0.30 anchor to 0.292, and the shots-on-target lift takes it to 0.329.
3. **Confidence floor** for unanchored props only: `player_ga` and `player_sot_ge` get
   $q \leftarrow \max(q, 0.15)$. It was chosen by leave-one-match-out CV over floors
   {0, 0.06, ..., 0.15}, and clean-set total Brier improved from 57.642 to 57.385
   (about +25.7 RBP). **Break-even:** raising $q$ from $a = 0.10$ to $b = 0.15$ changes
   expected Brier by $(b-p)^2 - (a-p)^2 = (b-a)(a+b-2p)$, which is negative iff
   $p > 0.125$. The low-priced player props cleared that. `player_scores` was exempt,
   because there you had edge.
4. **Lineup caps go last:** an out player gets ≤5%; a benched player gets ≤15% (score),
   ≤18% (goal or assist) or ≤35% (shots on target). These are applied as *ceilings*, so
   a mis-scraped "benched" caps a starter at a substitute's rate instead of zeroing
   them. Statuses older than 12 hours are ignored. This was learned from De Bruyne:
   benched, the book hadn't repriced, and the crowd lift pushed a stale starter price
   higher.
5. **Unresolved players** blend toward the crowd template (§2.10G step 6).

#### (h) Flat base-rate events: the crowd hug

For a whitelist of types (penalty or red, penalty awarded, a substitute scores) where
your model is a constant and the settled crowd barely varies (std ≤ 4 points), blend 60%
toward the crowd's typical level. *Example:* penalty or red goes from 0.339 to 0.362.

**Why it's right:** you have no question-specific signal, so you have two estimates of
the same base rate, and averaging them reduces error. Sitting at the field's own level
also locks in the $s^2$ bonus, instead of bleeding a fixed $(q - p^{\star})^2$ offset
every match. Offsides and fouls are **excluded** from the hug, because there the crowd
is *over*-priced.

#### (i) Global tail insurance: the crowd-proxy clamp

At submission, any **unbacked** price more than 25 points from the type's crowd prior is
pulled back *to the boundary*, keeping its direction. The prior must be stable to be
used: $n \ge 5$ and std ≤ 12, or $n \ge 3$ and std ≤ 6. Matchup-driven types such as
win (±17.6) fail that test and are left alone.

Two exemptions protect validated extremes:

- book-anchored markets, where the book *is* the crowd proxy;
- player-family *low* prices with no book, the fringe-player shots-on-target edge.

**Validation:** with the prior fitted on group-stage rows and applied to the round of
32, the clamp **fired zero times** on the fixed pipeline, so it costs nothing in normal
operation. Retroactively it catches exactly the historical catastrophes: the
both-teams-shots-on-target bug goes from 9 to 42, and its Brier from 0.83 to 0.34. It is
pure tail insurance, justified by the curve in §3.7(a).

#### (j) The LLM panel: narrow and gated

An offline panel of models reviews prices for things statistics can't see: confirmed
lineups, suspensions, a new penalty taker.

- **Live runs never call a model;** they read a cache.
- **Routing:** sharp markets are *excluded*. Proven-edge types are *shadow* (logged,
  never blended). Blending is allowed only on weak, model-blind types.
- **Blend weights** come from leave-one-match-out CV with the parsimony rule and a 5e-4
  improvement margin. The whole blend sits behind a master switch.
- **A verifier handles big disagreements** (|Δlogit| > 0.8). It follows the expert
  *only* on hard, verifiable evidence (confirmed XI, injury ruling), and only for player
  and lineup-driven props, with the move capped at 2.5 logits. Otherwise it keeps the
  model.
- **Opinions don't move prices.**

#### (k) Operational guards

- **Anchor-regression guard.** A run whose book prop line briefly vanished must not
  revert an anchored prop to the bare model (Diallo, 50 → 24). It now requires
  *ledger evidence* that the market was anchored on an earlier run. The first version
  lacked that condition and protected a stale mis-parse instead (DR Congo).
- **Divergence audit.** Any book-versus-model gap of 20+ points is flagged, because it
  signals either a data hole or a scope mismatch.
- **Name-join audit** each run.
- **Template audit** of every unparsed question.

### 3.10 The endgame: knockouts at 2×, the final at 3×

**The temptation.** §1.8 says extra variance helps someone who is behind, and the
multipliers are highest at the end.

**The simulation.** A block bootstrap over real settled match blocks gave:

| policy | closing-slate RBP |
|---|---|
| honest | **+481 ± 379** (P(losing ground) ≈ 10%) |
| blanket extremize | **+475 ± 760** |

Same expectation, twice the spread.

**Why the mean doesn't rise.** Extremizing a calibrated $p$ to $q$ costs $(q-p)^2$ per
question in expectation. Gains on right calls are offset by losses on wrong ones, and
the multipliers scale both. Variance is only worth buying where you have *structural*
edge.

**The policy ladder** (`endgame_policy.py`):

- Remaining multiplier-weighted questions: $W = (2n_{2\times} + 3n_{\text{final}})\cdot 15$.
- Projected cutoff: $C' = C + v_C\,W\cdot 0.88$, where $v_C$ is the target rank's pace
  per question, regressed by 0.88 for persistence.
- Needed rate: $v^{\text{need}} = (C' - \text{score})/W$, compared with your honest rate
  $v^{\text{hon}}$.
- Tiers:
  - $v^{\text{need}} \le v^{\text{hon}} + 0.5$: **honest**;
  - $\le +2$: dial 1.25;
  - $\le +4$: dial 1.5;
  - beyond that: 1.75, with a warning that it *lowers* expected rank.
- **The reading before the quarter-finals:** a score of 2,724 against 4,064 for 10th
  place needed +12.9 per question against an honest +2.0. The top 10 would have
  required a tail outcome, so **the posture stayed honest**, and it stayed honest
  through the final.

**The dial itself**, if armed with $\alpha > 1$, is $q = \sigma(\alpha\,\operatorname{logit}q)$,
and only on an allow-list of structural-edge types: win both halves, first goal in the
2H, margin exactly, odd total, goal after break 1, goalless at HT, corners and shots,
team shots on target, team corners, shots-on-target comparison, cards comparison. It is
never applied to player props (the crowd beat you there), fouls and offsides, anchored
goal markets, or card timing.

**Judgement staged in advance.** For the bronze match and the final, conditional rules
fired automatically at the closing tick only when a confirmed lineup matched. For
example: "if Kane is benched, the substitute-scores price moves up."

---

## Part 4 — What happened, honestly

### 4.1 Results

- **67th of 4,013** overall (top 1.7%) and **16th of 498** bots.
- **+3.2 relative Brier points per forecast**, better than 84% of forecasters, over
  **1,169** settled questions.
- Mean Brier **0.2236**, against 0.25 for a coin flip.

### 4.2 Calibration, recomputed for this guide (all 1,169 questions)

| submitted range | n | mean submitted | realised |
|---|---|---|---|
| 0.0–0.1 | 18 | 0.081 | 0.222 |
| 0.1–0.2 | 83 | 0.157 | 0.181 |
| 0.2–0.3 | 153 | 0.253 | 0.268 |
| 0.3–0.4 | 225 | 0.352 | 0.333 |
| 0.4–0.5 | 307 | 0.461 | 0.469 |
| 0.5–0.6 | 201 | 0.554 | 0.522 |
| 0.6–0.7 | 113 | 0.652 | 0.619 |
| 0.7–0.8 | 44 | 0.743 | 0.727 |
| 0.8–0.9 | 20 | 0.844 | 0.650 |
| 0.9–1.0 | 5 | 0.922 | 1.000 |

**Murphy decomposition:**

- reliability **0.0014**, which is tiny: the "elite calibration" rating;
- resolution 0.0212;
- uncertainty 0.2453.

The sum, $0.0014 - 0.0212 + 0.2453 = 0.2255$, matches the actual 0.2236 up to binning.

Two small tails are visible. The lowest bucket under-predicts (n = 18), and the
0.8–0.9 bucket over-predicts (n = 20). Both are small samples, and they are consistent
with the confident-low player problem and the over-confidence found in §3.7(b).

### 4.3 Where the points came from

On the 646 questions with crowd data, your Brier was 0.2244 and the consensus 0.2223.
The mean edge against the consensus was −0.0021 per question with a standard error of
0.0037 ($t = -0.57$): **statistically indistinguishable from consensus quality.**
Against the actual scoring baseline (the average individual) you were about +0.03 per
question ahead. So the honest attribution is:

- **Most of the +3.2 per forecast is the dispersion bonus $s^2 \approx 0.03$**, which
  you harvested by being as calibrated as the consensus. The market anchor gave you
  that cheaply, and the safety layers stopped you giving it back.
- **Genuine edge over the consensus** showed up where theory predicted it would: player
  props (+23), timing (+15), and team counts.
- **It was offset** by offsides (−62) and fouls, which were then pinned flat, and by
  anchored goal markets (−49 against the consensus). That last one is really the
  book-versus-crowd gap: on those markets you deliberately *were* the book.

That is a strong, defensible story: **you understood that the contest rewards
consensus-quality calibration, built the cheapest reliable path to it, and spent
modelling effort and risk only where an edge was measured.**

### 4.4 The lessons, in your own words

1. **Calibration over cleverness, especially under multipliers.** No deviation from a
   calibrated price without measured evidence that this kind of deviation has paid
   before.
2. **Bugs, not model error, drove the worst losses.** Misrouted questions and failed
   name joins cost more than any modelling choice, so loud audits are a statistical
   tool.
3. **When an estimate is unidentified, ship less than the fit says.** The crowd lift,
   the copula ρ and the confidence floor were all set at or below their fitted values,
   with caps doing the regularising.
4. **Know what is observable when.** The crowd is a post-lock calibration target, never
   a live signal.
5. **Measure value against the closing line.** It decided the player book weight (0.97)
   and justified keeping 0.85 on goals.

### 4.5 Weak points to own before an interviewer finds them

- The live edge gate compared you against a dulled copy of yourself (§3.3). Partly
  circular; superseded by real crowd data.
- The field model's $\lambda_f$ priors were judgement. Only one aggregate number (total
  RBP) could calibrate them, via $F \approx B_{\text{us}} + \text{RBP}/(100n)$, so only
  a global scalar was identifiable.
- Several constants are fixture-specific pins from the endgame (Messi's and Yamal's
  form rates, team shootout strengths). Defensible as evidence entry, but not a model.
- The 0.85 blend weight was fitted against FiveThirtyEight, not real closing lines. The
  CLV ledger was built to check it going forward.
- Many per-type adjustments rest on tiny samples (§3.7f). Your defence is that they are
  shrunk, cross-validated by match, and under-shifted.
- Card handlers ignore the referee: there was no live referee-assignment feed, and
  in-tournament referee factors didn't transfer.
- Corner and card book lines weren't available on your odds feed, so those families
  never had a market anchor.

---

## Part 5 — Interview drill

Short answers you should be able to give without notes. Derivations are in the sections
referenced.

**Q1. Prove the Brier score is strictly proper.**
$\mathbb E[(q-y)^2] = (q-p^{\star})^2 + p^{\star}(1-p^{\star})$, uniquely minimised at
$q = p^{\star}$. The regret is exactly $(q-p^{\star})^2$. (§1.2)

**Q2. Is *relative* Brier proper?**
For expected points, yes. Your report enters only through $-(q-p^{\star})^2$, because
the field term doesn't depend on you. For *rank*, no: if you're behind,
$\partial P(\text{win})/\partial\sigma > 0$, which rewards variance (Lichtendahl &
Winkler). (§1.5, §1.8)

**Q3. Why does copying the consensus earn positive points?**
$F = (\bar q - y)^2 + s^2$: the average of individual Brier scores exceeds the
consensus's Brier by the crowd's dispersion. By Jensen, the mean of a convex loss is at
least the loss at the mean. Here $s^2 \approx 0.03$, about +3 RBP per question at 1×.
(§1.4)

**Q4. Brier versus log score: what changes?**
Log score is unbounded, so a confident wrong call can cost infinitely much. Brier is
bounded at 1, which tolerates confident calls more. Both are proper. Under relative
Brier the asymmetry still exists, because the gain is capped at $F$ and the loss can
reach $1 - F$. (§1.6)

**Q5. What is the overround, and why Shin over normalising?**
Overround is $\sum 1/o_i - 1$. Proportional normalising keeps the favourite–longshot
bias. Shin models insider money and removes proportionally more margin from longshots;
the margin itself reveals the insider fraction $z$. (§2.5)

**Q6. Why average books in probability space, but blend model and market in logit
space?**
Books are correlated noisy measurements of one quantity, so the minimum-variance
combiner is an inverse-variance arithmetic mean. Model and market are diverse,
independent sources, and a linear pool of those is under-confident, so a geometric
(logit) pool is better. (§2.6)

**Q7. Why anchor intensities instead of each probability?**
Coherence and propagation. One $(\lambda_A,\lambda_B)$ pair drives every market, so
answers stay mutually consistent, and the market's information flows into questions no
book quotes (halves, windows, counts through $\mu(\lambda)$). (§2.8)

**Q8. Your inversion is overdetermined. What does the residual mean?**
There are 4 prices and 2 parameters. The residual is the part of the market's joint
view that independent Poissons with Dixon–Coles can't express, e.g. the draw coming out
1–2 points low. (§2.7)

**Q9. Derive the negative binomial and its variance.**
Use a Gamma–Poisson mixture: $X\mid G \sim \text{Pois}(\mu G)$ with $G \sim \Gamma(k,1/k)$.
Integrating gives the NB pmf. By total variance, $\operatorname{Var} = \mu + \mu^2/k$,
so $k = \mu^2/(\operatorname{Var}-\mu)$. (§2.10F)

**Q10. Two identical teams: what is P(A has more corners than B)?**
$(1 - P(\text{tie}))/2 < 1/2$. With mean 4.4 and $k = 5.3$, P(tie) = 0.109 and
P(A > B) = 0.446. Ties settle NO. (§2.10F)

**Q11. Derive P(A scores first).**
Superpose the two processes: the first event is A's with probability $\lambda_A/\Lambda$,
and some goal happens with probability $1 - e^{-\Lambda}$. So
$P = \frac{\lambda_A}{\Lambda}(1 - e^{-\Lambda})$. (§2.10B)

**Q12. P(total goals is odd)?**
$\mathbb E[(-1)^N] = e^{-2\Lambda}$, so $P(\text{odd}) = (1 - e^{-2\Lambda})/2 < 1/2$.
Dixon–Coles with $\rho < 0$ lowers it further. (§2.10C)

**Q13. Expected minutes or a minutes mixture: which overstates, and why?**
$1 - e^{-rM}$ is concave, so Jensen gives $\mathbb E f(M) \le f(\mathbb E M)$. Plugging
in expected minutes overstates; the example was 0.738 against 0.686. (§2.10G)

**Q14. State and prove Poisson thinning.**
Keeping each event independently with probability $\theta$ from a $\text{Pois}(\lambda)$
count gives $\text{Pois}(\theta\lambda)$: sum the binomial over $n$, and the tail
re-sums to $e^{(1-\theta)\lambda}$. (§2.10A)

**Q15. Why does conditioning on totals bias the goal minute profile?**
Goals create the score state that follows them, so the total isn't ancillary for the
shape, and the denominator treats an outcome-dependent path as fixed. Use the full
point-process likelihood with an offset instead. A synthetic-recovery test caught it.
(§2.10D)

**Q16. Why a sandwich covariance?**
Age-decay weights make it a pseudo-likelihood, so the information identity fails
($w^2 \ne w$). Then $\operatorname{Var}\hat\beta \approx H^{-1}MH^{-1}$ rather than
$H^{-1}$. Averaging predictions over parameter draws pulls thin-data extremes toward 50,
which is honest hedging. (§2.2)

**Q17. How did you validate edge with so little data?**
Per-question edge sd ≈ 0.093, so detecting a 0.01 edge needs about 345 questions and a
0.005 edge about 1,380. Per-type samples were 5–50, so nothing per type was identified.
You therefore used structure rather than fitting, cross-validation by match, parsimony,
no-regression gates, pre-registration, shrinkage to pooled values, and under-shifting.
(§3.7f, §3.8)

**Q18. Why leave one *match* out rather than one question?**
Questions within a match share $(\lambda_A,\lambda_B)$ and match-level randomness, so
they aren't independent. Cross-validating by question leaks, and an outlier match would
be memorised. (§3.8)

**Q19. You were well behind the top 10 late in the tournament. Why not gamble?**
The bootstrap gave the same mean at twice the spread (+481 ± 379 against +475 ± 760).
Extremizing a calibrated forecast costs $(q-p)^2$ in expectation, and the multipliers
scale losses too. The rate needed for the top 10 (+12.9 per question against +2.0
honest) meant only a tail outcome would get there, and buying variance would lower
expected rank. (§3.10)

**Q20. The crowd beat the book on player props. Why not just submit the crowd?**
It's invisible before lock. You estimated a stable crowd-minus-book premium, but its
magnitude was unidentified (a monotone curve, $p \approx 0.065$, direction 15–14), so
you applied half of it with a 6-point cap. (§3.9g)

**Q21. Why can a constant beat your fouls model?**
If the model's per-match variation is uncorrelated with the outcome, it adds variance
without resolution. Each question then pays $(q - p^{\star})^2$ extra relative to the
right constant. The fix is to shrink hard ($s = 0.15$) toward the symmetric 0.50.
(§3.7d, §3.9e)

**Q22. Isn't clamping toward the crowd giving up edge?**
Only for deviations of 25+ points, which ran at −0.079 per question and were almost all
bugs. Validated edges (book-anchored, fringe-player lows) are exempt, and on the fixed
pipeline the clamp fired zero times out of sample. (§3.7a, §3.9i)

**Q23. How do you estimate the field without seeing it?**
Use a structural model $f = \sigma(\lambda_f\operatorname{logit}p)$ with family priors.
Calibrate one global level from aggregate RBP via
$F \approx B_{\text{us}} + \text{RBP}/(100n)$. Later, use harvested post-lock crowd data
as the grader. (§3.2, §3.6)

**Q24. Extremize the probability or the deviation?**
The deviation from the field. Extremizing raw probabilities on markets where you *equal*
the field is a zero-edge gamble that also raises your Brier. (§3.2)

**Q25. What is a well-calibrated forecaster's expected Brier?**
$\mathbb E[p^{\star}(1-p^{\star})]$, which is the uncertainty term minus resolution. Here
the base rate was ≈ 0.43, giving uncertainty 0.245; resolution 0.021 brought it down to
about 0.224. (§4.2)

**Q26. How much of the season total is luck?**
$0.093\sqrt{1169}\times100 \approx 318$ RBP of standard deviation at 1×. That is large
compared with neighbouring rank gaps, so judge process by calibration and by edge
against the consensus, not by rank alone. (§1.8)

**Q27. What would you do differently?**
Harvest the crowd from day one, since it's the only true grader. Get a feed with
corner and card lines. Build the parse-and-route audit before the model. Pre-register
more of the gate tests. Model referees with career-level rates.

---

## Part 6 — Cheat sheet

| quantity | value | source |
|---|---|---|
| Brier regret | $(q-p^{\star})^2$ | derivation |
| field identity | $F = (\bar q-y)^2 + s^2$ | Jensen |
| dispersion bonus | $s^2 \approx 0.03$ ($s \approx 17$ points) | crowd harvest: consensus ≈ 0.222, average individual ≈ 0.252–0.256 |
| stage multipliers | 1× group, 2× knockouts, 3× final | rules |
| Elo $K$ | WC 60 / continental 50 / qualifiers 40 / minor 30 / friendly 20; home +100 | `elo.py` |
| Elo → goals | $\beta_1 \approx 0.35$ (100 Elo ≈ ×1.42 goals) | goal model fit |
| half-life | 730 days (1, 2, 4 years all flat) | backtest |
| Dixon–Coles $\rho$ | ≈ −0.03 to −0.06 | fit |
| market weight $w$ | 0.85 (flat 0.65–1.0) | 84 fixtures against 538 |
| player book weight | 0.97 ($= 1 - 0.2(1-w)$) | 51 anchored props: 0.250 / 0.206 / 0.202 |
| second-half goal share | 0.556 | 47k goals |
| hydration breaks | minutes 22 and 67 | platform |
| corners var/mean | 1.42 | StatsBomb |
| corners–shots coupling | ρ = 0.30 (measured 0.41) | 314 matches |
| shrinkage pseudo-minutes | 270 (shots on target), 540 (goals, G+A) | backtest |
| club → country | τ = 0.80 | settled props |
| minutes mixture | (0.80, 80′), (0.12, 25′), (0.08, 0′) | judged |
| ET tempo / pens tilt | 0.35 / 0.20 | 839 ET, 678 shootouts |
| P(penalty) / P(red) | 0.293 / 0.065 | 6 tournaments |
| field sharpness priors | goal 0.85, player 0.45, count 0.40, timing 0.35 | judged |
| live edge gate | n ≥ 20, margin 0.005 → κ = 1.3 (timing only) | `update_edge_gate.py` |
| risk gate | offsides/fouls s = 0.15 (bases 0.40/0.50), cards 0.5, both-teams SoT 0.3 | crowd data |
| confidence floor | 0.15 on `player_ga`, `player_sot_ge` (unanchored) | leave-one-match-out CV |
| crowd lift | 0.5 × premium, cap 6 points, only below 0.85 | under-shift by design |
| crowd hug | 60% toward crowd where std ≤ 4, n ≥ 3 | flat event types |
| crowd clamp | ±25 points from the crowd prior; book-anchored and player lows exempt | risk curve |
| endgame bootstrap | honest +481 ± 379 against blanket +475 ± 760 | block bootstrap |
| result | 67/4,013; 16/498 bots; +3.2/forecast; 1,169 settled; Brier 0.2236; reliability 0.0014 | platform + recomputation |

---

## Appendix — Where everything lives in the code

| topic | file |
|---|---|
| Elo | `src/jumpprob/features/elo.py` |
| goal model, Dixon–Coles grid | `src/jumpprob/models/goal_model.py` |
| sandwich covariance | `src/jumpprob/models/uncertainty.py` |
| de-vig (basic/power/Shin), consensus | `src/jumpprob/models/devig.py` |
| implied intensities, log blends | `src/jumpprob/models/market_anchor.py` |
| odds feed, player-prop de-vig, name join | `src/jumpprob/ingest/odds_api.py` |
| negative-binomial count model | `src/jumpprob/models/count_model.py` |
| minute profile and state multipliers | `src/jumpprob/models/dynamics.py` |
| Monte-Carlo engine | `src/jumpprob/simulate/mc_engine.py` |
| grid market readers | `src/jumpprob/simulate/markets.py` |
| Platt, integer submission | `src/jumpprob/models/calibration.py` |
| field model, extremizer, risk gate, confidence floor | `src/jumpprob/models/strategy.py` |
| question handlers | `src/jumpprob/submit/handlers/{goals,counts,players,events}.py` |
| the pricing ladder | `src/jumpprob/submit/runner.py` |
| every tuned constant, with its justification | `src/jumpprob/submit/constants.py` |
| crowd clamp, anchor-regression guard, divergence audit | `src/jumpprob/submit/audit.py` |
| LLM panel routing and gated blend | `src/jumpprob/agents/panel.py` |
| live run: three passes, clamp, ledger, submit | `scripts/run_matchday_odds.py` |
| nightly refits | `scripts/daily_learn.py` |
| edge gate and base rates | `scripts/update_edge_gate.py` |
| settled + crowd dataset | `scripts/build_settled_dataset.py` |
| confidence-floor fit (leave-one-match-out CV) | `scripts/fit_confidence.py` |
| crowd prior / player crowd premium | `scripts/fit_crowd_prior.py`, `scripts/fit_player_crowd_premium.py` |
| per-family Platt | `scripts/fit_live_calibration.py` |
| pre-registered shadow tests | `scripts/shadow_decalib.py` |
| endgame policy ladder | `scripts/endgame_policy.py` |
| Bot B sizing | `scripts/submit_gigabot2.py` |
| full derivations reference | `docs/METHODS.md`, `docs/MARKET-FIRST-ARCHITECTURE.md` |
