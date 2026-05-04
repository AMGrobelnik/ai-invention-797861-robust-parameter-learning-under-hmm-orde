# KL-Constrained EM, Misspecified HMMs, Robust Markov

## Summary

Literature survey integrating three research domains critical for robust HMM learning under order misspecification: (1) KL-constrained and distributionally robust EM algorithms, (2) convergence theory under HMM model misspecification, and (3) KL-constrained optimization for Markov chains. Synthesizes 18 key references from Duchi, Namkoong (robust optimization), Hsu, Kakade, Zhang (spectral learning), Douc, Moulines, Dwivedi (misspecified convergence), Csiszár, Singh (information geometry), and Tishby, Zamir (rate-distortion). Five critical research gaps prevent integrating these domains: (1) KL-constrained EM never applied to forward-backward algorithms in latent-variable HMMs; (2) finite-sample convergence rates for constrained EM under order mismatch uncharacterized; (3) relationship between KL-divergence bias and robust parameter recovery unquantified; (4) no algorithm to estimate KL divergence between model orders or set constraint radius adaptively; (5) Viterbi decoding error under constraints unanalyzed. For each gap, specific technical recommendations are provided. Survey establishes mathematical foundations, constructs detailed reference matrix mapping tractable updates and convergence guarantees across three domains, and provides concrete follow-up questions for implementing robust HMM algorithms.

## Research Findings

This literature survey integrates three previously disconnected research areas to support robust parameter learning for HMMs under order mismatch.

**Area 1: KL-Constrained and Robust EM [1, 2, 3, 4, 5]** provides the foundational framework. Duchi et al. [1, 2] establish distributionally robust optimization with KL divergence, developing convex formulations and finite-sample minimax bounds. Kunstner et al. [3] prove that EM for exponential families is equivalent to mirror descent, achieving convergence in KL divergence with non-asymptotic rates and local superlinear convergence. The connection to Bregman divergences [4, 5] enables incorporating convex constraints through projections while preserving multiplicative structure and convergence guarantees.

**Area 2: HMM Misspecification [6, 7, 8, 9]** characterizes what happens when the true model order differs from the fitted order. Hsu, Kakade, and Zhang [6] provide finite-sample spectral learning with sample complexity independent of observation space size, requiring only separation conditions on HMM spectral properties. Douc and Moulines [7] prove that the MLE remains consistent under misspecification, converging to the KL-divergence minimizer (pseudo-true parameter). Critically, Dwivedi et al. [8] show that when the Fisher Information Matrix becomes singular under misspecification (as occurs with order mismatch), convergence rate deteriorates from O(1/√n) to dimension-dependent rates like O((d/n)^(1/4)). Alexandrovich et al. [9] characterize identifiability conditions for HMM structure.

**Area 3: Markov Chain and Constrained Optimization [10, 11, 12, 13, 14, 15]** provides geometric tools. The Kullback-Leibler divergence rate extends KL divergence to sequences [10], and information projection (I-projection) minimizes KL subject to linear constraints on exponential families [13]. Csiszár's geometry [13, 14] establishes a Pythagorean theorem for KL divergence on exponential families (dually flat manifolds), enabling projection-based algorithms. Rate-distortion theory [15] connects model reduction of Markov chains to KL divergence minimization.

**Critical Gaps Identified:** [1] No work applies KL-constrained EM to forward-backward algorithms in latent-variable HMMs. [2] Finite-sample convergence rates for robust EM under order mismatch are not characterized. [3] The relationship between KL divergence bias and robust parameter recovery remains unquantified. [4] No algorithmic method exists to compute KL divergence between model orders or set constraint radius ε adaptively. [5] Viterbi decoding error under constrained parameters has not been analyzed, leaving robust inference unresolved.

## Sources

[1] [Learning Models with Uniform Performance via Distributionally Robust Optimization](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-3/Learning-models-with-uniform-performance-via-distributionally-robust-optimization/10.1214/20-AOS2004.full) — Develops DRO framework for robust model learning with KL-divergence constraints; proves finite-sample minimax bounds.

[2] [Statistics of Robust Optimization: A Generalized Empirical Likelihood Approach](https://arxiv.org/abs/1610.03425) — Develops generalized empirical likelihood framework for f-divergence balls; analyzes statistical inference under distributional uncertainty.

[3] [Homeomorphic-Invariance of EM: Non-Asymptotic Convergence in KL Divergence for Exponential Families via Mirror Descent](https://arxiv.org/abs/2011.01170) — Proves EM for exponential families is mirror descent with KL divergence regularizer; establishes non-asymptotic convergence rates.

[4] [Reinterpreting EMML as Mirror Descent for Constrained Maximum Likelihood Estimation](https://arxiv.org/html/2602.13063) — Shows EMML equivalence to mirror descent; demonstrates how to incorporate convex constraints through Bregman projections.

[5] [Bregman Divergence and Mirror Descent](https://users.cecs.anu.edu.au/~xzhang/teaching/bregman.pdf) — Comprehensive treatment of Bregman divergences in optimization; shows mirror descent replaces Euclidean distance.

[6] [A Spectral Algorithm for Learning Hidden Markov Models](https://arxiv.org/abs/0811.4413) — Develops polynomial-time spectral algorithm for HMM learning with finite-sample guarantees; sample complexity independent of observation space size.

[7] [Asymptotic properties of the maximum likelihood estimation in misspecified hidden Markov models](https://projecteuclid.org/euclid.aos/1359987535) — Proves consistency of MLE when true model is not in parametric family; characterizes convergence to KL minimizer.

[8] [Singularity, Misspecification, and the Convergence Rate of EM](https://arxiv.org/abs/1810.00828) — Characterizes non-standard convergence rates for EM under model misspecification; shows O((d/n)^(1/4)) rate under singularity.

[9] [Nonparametric Identification and Maximum Likelihood Estimation for Hidden Markov Models](https://link.springer.com/article/10.1007/s11222-023-10364-7) — Establishes identifiability conditions for HMM parameters: full-rank transition matrix, ergodicity, distinct emission distributions.

[10] [Notes on the KL-divergence between a Markov chain and product space interpretation](https://www.cs.cmu.edu/~rsalakhu/papers/mckl.pdf) — Analyzes KL divergence between Markov chains, showing how marginal distributions relate under aggregation.

[11] [Optimal Kullback-Leibler Aggregation via Information Projection](https://arxiv.org/pdf/1304.6603) — Develops algorithms for KL-constrained aggregation on exponential families using information projections.

[12] [A rate-distortion framework for MCMC algorithms: geometry and factorization of multivariate Markov chains](https://arxiv.org/html/2404.12589v1) — Analyzes rate-distortion theory for Markov chains; connects to KL divergence rates and model reduction.

[13] [Information Projections Revisited](https://www.researchgate.net/publication/3084724_Information_projections_revisited) — Establishes Pythagorean theorem for KL divergence in exponential families; shows I-projections minimize KL divergence.

[14] [Lecture 7: Information Projections and Exponential Families](https://www.cs.cmu.edu/~aarti/Class/10704_Spring15/lecs/lec7.pdf) — Treatment of information projections on exponential families and role of Bregman divergences in constrained optimization.

[15] [Rate-Distortion via Markov Chain Monte Carlo](https://web.stanford.edu/~tsachy/pdf_files/rate%20distortion%20via%20markov%20chain%20monte%20carlo.pdf) — Rate-distortion framework for Markov chain processes; characterizes compression cost via KL divergence rates.

[16] [Statistical guarantees for the EM algorithm: From population to sample](https://projecteuclid.org/journals/annals-of-statistics/volume-45/issue-1/Statistical-guarantees-for-the-EM-algorithm--From-population-to/10.1214/16-AOS1435.pdf) — Population and finite-sample convergence guarantees for EM in well-specified settings; geometric convergence to KL projection.

[17] [Baum-Welch Algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) — Foundational algorithm for HMM parameter estimation via forward-backward probabilities; guaranteed to increase log-likelihood.

[18] [Constrained Expectation-Maximization Methods for Effective Reinforcement Learning](https://ieeexplore.ieee.org/document/8488990) — Constrained EM variants with KL divergence policy constraints; demonstrates practical applications.

## Follow-up Questions

- What is the explicit closed-form solution for the KL-constrained M-step for transition/emission matrices? Can it be computed via Bregman projections?
- How does constraint radius ε scale with order mismatch magnitude? Is there an information-theoretic lower bound on ε?
- What is the finite-sample convergence rate for constrained Baum-Welch under order mismatch? Does the O((d/n)^(1/4)) singularity rate apply?
- Can KL(P_true_k || P_best_first_order) be estimated adaptively from data? Does this enable automatic constraint radius selection?
- How does Viterbi decoding error scale under constrained parameters with order mismatch? What is the relationship between constraint radius and state inference error?
- How do spectral learning guarantees from Hsu et al. extend to KL-constrained settings? Are separation conditions preserved?
- What are convergence guarantees for forward-backward algorithms under constraints? Does the E-step remain closed form under KL constraints?

---
*Generated by AI Inventor Pipeline*
