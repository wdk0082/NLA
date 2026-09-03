# Metrics for NLA Language Alignment, Claim Support, and Claim Importance

## 1. Overview

Consider a natural-language autoencoder (NLA) that verbalizes an internal activation of a target language model.

Let

$$
x = \text{input text},
\qquad
h = h(x) = \text{target activation},
\qquad
z = \text{NLA verbalization}.
$$

Let $R$ be the activation reconstructor, so that $R(z)$ approximates $h$. Let $F$ be the frozen downstream part of the target language model. Then

$$
p = F(h),
\qquad
\hat p(z) = F(R(z)),
$$

where $p$ is the target model's original output distribution and $\hat p(z)$ is the output distribution obtained from the reconstructed activation.

The evaluation separates three questions:

1. **Language alignment:** Do humans and the NLA treat the same sentence pairs as semantically equivalent?
2. **Claim support:** Is a claim consistent with the input, the activation, and the model output?
3. **Claim importance:** How much does a claim contribute to reconstructing the activation and output?

---

## 2. Reconstruction losses

Define the normalized activation-reconstruction loss

$$
\mathcal L_h(z)
=
\frac{\|h-R(z)\|_2^2}{V_h},
$$

where $V_h$ is a normalization constant such as the total activation variance.

Define the output-reconstruction loss

$$
\mathcal L_o(z)
=
D_{\mathrm{KL}}\!\left(p\,\|\,\hat p(z)\right).
$$

For a claim $c$ appearing in $z$, define two modified verbalizations:

- $z^{\neg c}$: replace $c$ by its contradiction while changing as little else as possible;
- $z^{-c}$: remove $c$ without replacing it.

Contradiction measures whether a claim is **supported**. Deletion measures whether a claim is **important**.

---

## 3. Full metric table

| Question | Metric | Definition | Interpretation |
|---|---|---|---|
| **1. Human–NLA language alignment** | **Steganography rate** | $$\epsilon_{\mathrm{steg}}=\Pr(N=0\mid H=1)$$ | Human-equivalent sentences are treated differently by the NLA. |
|  | **Semantic aliasing rate** | $$\epsilon_{\mathrm{alias}}=\Pr(N=1\mid H=0)$$ | Human-different sentences are treated as equivalent by the NLA. |
| **2. Claim consistency and support** | **Input-text consistency** | $$S_x(c)=\Pr(\mathrm{entail}\mid x,c)-\Pr(\mathrm{contradict}\mid x,c)$$ | Whether $c$ agrees with the observable input $x$. This is an input-based proxy, not directly observable ground-truth truth. |
|  | **Activation support** | $$S_h(c)=\mathcal L_h(z^{\neg c})-\mathcal L_h(z)$$ | Whether $c$, rather than its negation, better represents the target activation $h$. |
|  | **Output support** | $$S_o(c)=\mathcal L_o(z^{\neg c})-\mathcal L_o(z)$$ | Whether $c$, rather than its negation, better reconstructs the target model's output. |
| **3. Claim importance** | **Activation-reconstruction importance** | $$I_h(c)=\mathcal L_h(z^{-c})-\mathcal L_h(z)$$ | How much $c$ contributes to reconstructing $h$. Under normalized MSE, this is the corresponding drop in explained activation variance. |
|  | **Output-reconstruction importance** | $$I_o(c)=\mathcal L_o(z^{-c})-\mathcal L_o(z)$$ | How much $c$ contributes to reconstructing the target model's output. |

---

## 4. Human–NLA language alignment

For two verbalizations $z$ and $z'$, let

$$
H(z,z')\in\{0,1\}
$$

denote whether humans regard them as semantically equivalent.

Define NLA equivalence operationally by

$$
N(z,z')
=
\mathbf 1\!\left[
\frac{\|R(z)-R(z')\|_2^2}{V_h}
\leq \tau_N
\right],
$$

where $\tau_N$ is an equivalence threshold.

The two asymmetric alignment errors are then

$$
\epsilon_{\mathrm{steg}}
=
\Pr(N=0\mid H=1),
$$

and

$$
\epsilon_{\mathrm{alias}}
=
\Pr(N=1\mid H=0).
$$

These probabilities are defined relative to a specified distribution of sentence pairs.

---

## 5. Claim support versus claim importance

The two interventions answer different questions:

$$
\boxed{
\text{Contradiction } z^{\neg c}
\quad\Longrightarrow\quad
\text{Is the claim supported?}
}
$$

$$
\boxed{
\text{Deletion } z^{-c}
\quad\Longrightarrow\quad
\text{Does the claim matter?}
}
$$

Each claim therefore has a three-level support profile

$$
\operatorname{support}(c)
=
\bigl(S_x(c),S_h(c),S_o(c)\bigr),
$$

and a two-level importance profile

$$
\operatorname{importance}(c)
=
\bigl(I_h(c),I_o(c)\bigr).
$$

A claim may be inconsistent with the input text while still being supported by the target activation or by the target model's output. Therefore, input inconsistency should not be conflated with activation-level or output-level unfaithfulness.
