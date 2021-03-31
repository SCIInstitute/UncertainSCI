# Defining random parameters

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
<link rel="stylesheet" href="_static/css/main.css">

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

Authors:  


## Overview

UncertainSCI propagates randomness in input parameters to the distribution of the model output. In order to accomplish this, a probability distribution on the input random parameters must be specified. This tutorial describes how to create random parameters for use in UncertainSCI.

Let $\mathbf{p}$ be a $d$-dimensional random parameter,
\begin{align}
\mathbf{p} = (p_1, \ldots, p_d),
\end{align}
and assume that all parameters are independent. In order to specify the distribution of $\mathbf{p}$, we specify the distribution for each parameter $p_i$ individually for $i = 1, \ldots, d$.

## Types of distributions

### Paramtric distributions

| Distribution type | UncertainSCI class | Distribution parameters |
|-------------------|--------------------|-------------------------|
| Beta distribution | `BetaDistribution` | Shape `alpha`$\in (0,\infty)$ and `beta`$\in(0,\infty)$ |
| Normal distribution | `NormalDistribution` | Mean `mu`$\in (-\infty,\infty)$ and `variance`$\in(0,\infty)$ |
| Exponential distribution | `ExponentialDistribution` | Mean `mu`$\in (-\infty,\infty)$ and `variance`$\in(0,\infty)$ |

### Non-parametric distributions
TODO

## Examples

The following instantiates a uniform distribution on $[0,1]$, 
```
from UncertainSCI.distributions import BetaDistribution

p = BetaDistribution(alpha=1, beta=1)
```

One can generate samples from this distribution. For example, 

```
samples = p.MC_samples(100)
```

returns 100 random samples from this distribution.
