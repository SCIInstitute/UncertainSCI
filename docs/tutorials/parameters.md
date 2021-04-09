# Defining random parameters

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
<link rel="stylesheet" href="_static/css/main.css">

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

## Overview

UncertainSCI propagates randomness in input parameters to the distribution of the model output. In order to accomplish this, a probability distribution on the input random parameters must be specified. This tutorial describes how to create random parameters for use in UncertainSCI.

Let \\(\mathbf{p}\\) be a \\(d\\)-dimensional random parameter,
\begin{align}
\mathbf{P} = (P_1, \ldots, P_d),
\end{align}
and assume that all parameters are independent. UncertainSCI specifies the joint distribution of \\(\mathbf{P}\\) by defining univariate distributions for each parameter \\(P_i\\) for \\(i = 1, \ldots, d\\). There are two main ways to specify these distributions:

- A univariate distribution for each \\(P_i\\) can be individually specified, and the joint distribution can be built from these univariate ones using the `TensorialDistribution` class.
- If \\(P_i\\) for every \\(i = 1, \ldots, d\\) each have distributions from the same parametric family (for example, if each \\(P_i\\) has an exponential distribution), then the joint distribution for \\(P\\) can be built from one call to the parametric family class.

We describe below first how to create univariate distributions of various types, and this is followed by examples of how to build joint (multivariate) distributions.

## Types of distributions

If \\(P\\) is a scalar continuous random variable, we let \\(f_P\\) denote its probability density function (pdf). When \\(P\\) is discrete, we will still write a density \\(f_P\\) using Dirac (delta) masses located on the discrete support points of \\(P\\). In particular, \\(\delta_y(p)\\) denotes the Dirac delta, a function of \\(p\\), centered at \\(p = y\\).

### Beta Distribution

Let \\(P\\) be a random variable with Beta distribution, having shape parameters \\(\alpha\in (0,\infty)\\) and \\(\beta\in(0,\infty)\\). The pdf for \\(P\\) is 

\begin{align}
  f_P(p) &= \frac{ p^\alpha (1-p)^\beta}{B(\alpha,\beta)}, & p &\in (0, 1),
\end{align}
where \\(B(\cdot,\cdot)\\) is the Beta function, which is a normalization constant ensuring that \\(f_P\\) is a pdf.

The shape parameters \\(\alpha\\) and \\(\beta\\) dictate how the mass of \\(P\\) concentrates toward/away from the endpoints \\(0\\) and \\(1\\). Specializations of the Beta distribution include:
- Uniform distribution on \\([0, 1]\\): \\(\alpha = \beta = 1\\)
- Arcsine distribution on \\([0, 1]\\): \\(\alpha = \beta = \frac{1}{2}\\)
- Wigner semicircle distribution on \\([0, 1]\\): \\(\alpha = \beta = \frac{3}{2}\\)

Beta distributions are instantiated in UncertainSCI using the `distributions.BetaDistribution` class, which takes as inputs the shape parameters `alpha` and `beta`, corresponding to \\(\alpha\\) and \\(\beta\\), respectively. For example, the following creates a parameter `P` having Beta distribution with \\((\alpha,\beta) = (1,1)\\):

```
from UncertainSCI.distributions import BetaDistribution

P = BetaDistribution(alpha=1, beta=1)
```

### Normal distribution

Let \\(P\\) be a random variable with a normal distribution, having mean \\(\mu\in \mathbb{R}\\) and variance \\(\sigma^2\in(0,\infty)\\). The pdf for \\(P\\) is 

\begin{align}
  f_P(p) &= \frac{ 1}{\sigma \sqrt{2 \pi} } \exp \left( -\frac{(p-\mu)^2}{2 \sigma^2} \right), & p &\in \mathbb{R}.
\end{align}

Normal distributions are instantiated in UncertainSCI using the `distributions.NormalDistribution` class, which takes as inputs the distribution statistics `mean` and `cov`, corresponding to \\(\mu\\) and \\(\sigma^2\\), respectively. For example, the following creates a parameter `P` having normal distribution with \\((\mu,\sigma^2) = (1,3)\\):

```
from UncertainSCI.distributions import NormalDistribution

P = NormalDistribution(mean=1, cov=3)
```

### Exponential distribution

Let \\(P\\) be a random variable with an exponential distribution, having minimum \\(p_0 \in \mathbb{R}\\) and shape parameter \\(\lambda \in (0, \infty)\\). The pdf for \\(P\\) is 

\begin{align}
  f_P(p) &= \lambda e^{-\lambda (p - p_0)}, & p &\in [p_0, \infty).
\end{align}

Exponential distributions are instantiated in UncertainSCI using the `distributions.ExponentialDistribution` class, which takes as inputs the parameters `loc` and `lbd`, corresponding to \\(p_0\\) and \\(\lambda\\), respectively. For example, the following creates a parameter `P` having exponential distribution with \\((p_0,\lambda) = (-1,2)\\):

```
from UncertainSCI.distributions import ExponentialDistribution

P = ExponentialDistribution(loc=-1, lbd=2)
```

### Discrete uniform distribution

Let \\(P\\) be a random variable with a discrete uniform distribution, having \\(n \in \mathbb{N}\\) equally-spaced support points on \\([0, 1]\\).  The pdf for \\(P\\) is 

\begin{align}
  f_P(p) &= \sum_{j=1}^n \frac{1}{n} \delta_{\frac{j-1}{n-1}}(p)
\end{align}

Discrete uniform distributions are instantiated in UncertainSCI using the `distributions.DiscreteUniformDistribution` class, which takes as inputs the parameter `n`, corresponding to the number of support points \\(n\\). For example, the following creates a parameter `P` having discrete uniform distribution with \\(n = 17\\):

```
from UncertainSCI.distributions import DiscreteUniformDistribution

P = DiscreteUniformDistribution(n=17)
```

### Non-parametric distributions

Coming soon....
