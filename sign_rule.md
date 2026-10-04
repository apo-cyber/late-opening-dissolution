# Staged acceptance of opening-type products: the sign rule for the threshold depth and the four-class expression

Propositions behind the section "The direction of the error" of the paper and Supplementary Section S4. The numerical checks are in `sign_rule.py`, with output in `output/sign_rule.txt`. Notation follows the paper: the individual limit R (Interpretation 2; equal to Q + 5 of stage S1) and Q (Interpretation 1).

## Setting: the two-point limit model

A unit value X lies, with probability ε, at a low value a (a unit that has not opened), and with probability 1 − ε in an upper cluster N(m, s²). Write the gap as D = m − a, the relative depth of a threshold c (a < c < m) as δ(c) = (m − c)/D, and κ = s/D. Write F(c) = P(X < c) for the true fraction and G(c) for the fraction under the normal distribution with the same mean and SD. The mean and variance of that normal distribution are

μ = m − εD, σ² = D²{(1 − ε)κ² + ε(1 − ε)}.

## Proposition 1 (sign rule for the threshold depth)

As κ → 0, F(c) = ε and G(c) = Φ(−(δ − ε)/√(ε(1 − ε))). Hence

**G(c) > F(c) ⇔ δ(c) < δ\*(ε) = ε + z_{1−ε} √(ε(1 − ε))** (z_u is the u quantile of the standard normal distribution).

Below a shallow threshold (close to the upper cluster) the normal calculation overstates the fraction; below a deep threshold it understates it. For fixed δ and ε → 0, x = (δ − ε)/√(ε(1 − ε)) → ∞ and G(c) ~ φ(x)/x (Mills' ratio), so G/F → 0, at a rate of order exp(−δ²/(2ε)). Also δ\*(ε) → 0 as ε → 0 (δ\* is of the order of √(2ε ln(1/ε))).

*Proof*: μ and σ follow from the mean and variance of the two-point distribution. Since a < c < m, F(c) = ε. Substituting into G(c) = Φ((c − μ)/σ) and simplifying gives the form above. G > F ⇔ −(δ − ε)/√(ε(1 − ε)) > Φ⁻¹(ε) = −z_{1−ε}. The limit follows from Mills' ratio Φ(−x) ~ φ(x)/x. ∎

For κ > 0, δ\* increases slightly (numerically, at ε = 0.03: 0.351, 0.353, 0.364, 0.400 for κ = 0, 0.02, 0.05, 0.10).

| ε | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|---|
| δ\*(ε) (κ → 0) | 0.187 | 0.241 | 0.308 | 0.351 | 0.408 | 0.484 | 0.537 |

Example (a = 0, m = 98, nominal specification R = 75 % and Q = 70 % at 30 min): δ(R) = 0.235, δ(Q − 15) = 0.439, δ(Q − 25) = 0.541. With ε = 0.03, δ\* = 0.351, so **the normal calculation overstates the fraction below R and understates the fractions below Q − 15 and Q − 25**.

## Proposition 2 (sign of the error under Interpretation 2)

The probability of an OOS result under Interpretation 2 is a function h(p) of p = F(R) alone, and h is increasing in p (the decision is a rule monotone in the number of units below R). Hence the sign of the error of the normal calculation, h(G(R)) − h(F(R)), is the sign of G(R) − F(R), that is, it is decided by δ(R) versus δ\*(ε) as in Proposition 1. In the two-point limit p = ε and h(ε) ≈ 200ε³ (0.0002 at ε = 0.01; 0.0044 at 0.03, where the approximation gives 0.0054).

## Proposition 3 (Interpretation 1, two-cluster limit)

Let a < Q − 25, and let the upper cluster lie at or above Q + 5 almost surely (s → 0, m > Q + 5). Then

**P(failing Interpretation 1) → 1 − (1 − ε)⁶.**

If none of the first 6 units is unopened, the lot passes at S1; if at least one is, neither S2 (no unit among 12 below Q − 15) nor S3 (no unit among 24 below Q − 25) can be met. Further testing cannot rescue the lot. Under the normal distribution with the same mean and SD, on the other hand, failure at S3 requires "mean < Q", "3 or more units below Q − 15" or "1 or more units below Q − 25", so by the union bound

P_N(fail) ≤ Φ(−√24 (μ − Q)/σ) + 24·G(Q − 25) + C(24, 3)·G(Q − 15)³,

and each term on the right tends to 0 faster than any power as ε → 0 (μ → m > Q, σ = D√(ε(1 − ε)) → 0, and G as in Proposition 1). Since the true probability of failing is ~6ε, **the ratio P_N/P → 0: the normal calculation misses the failures of early ageing**. ∎

## Corollary (four-class expression)

As long as the mean clauses (the "mean ≥ Q" of S2 and S3) are met, the probability of passing Interpretation 1 does not depend on the distribution of unit values except through the probabilities of four classes,

pA = P(X ≥ Q + 5), pB = P(Q − 15 ≤ X < Q + 5), pC = P(Q − 25 ≤ X < Q − 15), pD = P(X < Q − 25):

P(pass) = pA⁶ + [(pA + pB)¹² − pA⁶(pA + pB)⁶] + Σ_{(c₁,c₂,c₃)} w(c₁)·b₆(c₂)·b₁₂(c₃),

where b_n(c) = C(n, c) pC^c (pA + pB)^{n−c}, w(c₁) = b₆(c₁) for c₁ ≥ 1, w(0) = (pA + pB)⁶ − pA⁶, and the sum runs over the triples with c₁ + c₂ ≥ 1 and c₁ + c₂ + c₃ ≤ 2 (c₁, c₂, c₃ are the numbers of class-C units among units 1–6, 7–12 and 13–24; any class-D unit makes S3 fail). The first term is passing at S1, the second failing S1 and passing S2, the third failing S2 and passing S3.

**Meaning**: the whole error of the normal calculation comes from misestimating the two deep classes (pC, pD). By Proposition 1, for a two-cluster lot the normal calculation underestimates pC and pD by orders of magnitude.

## Numerical checks (`output/sign_rule.txt`)

- The unit model (the 9 states of the three ageing types plus 400 random states) was fitted by two points (boundary between the lower and upper clusters at Q − 20). The sign predicted from (ε, δ) was correct in 164/164 states at the individual limit R and 163/163 at Q − 15 (states in which the true and normal fractions differed by less than 20 % were excluded).
- Probability of failing Interpretation 1: simulation (200,000 lots) and the four-class expression agreed within 0.005 in all 9 states (spread opening, p 0.05: 0.176 / 0.172; p 0.20: 0.688 / 0.685). The limit 1 − (1 − pD)⁶ is lower (0.150 and 0.572), because partly opened units (classes B and C) cause failures at S1 and S2.
- Interpretation 2: h(F(R)) agrees with simulation (as it must, by definition).
- Asymptotic form of Proposition 1: the Mills-ratio approximation is within 5 % for ε ≤ 0.01 and within 13 % at ε = 0.03.
