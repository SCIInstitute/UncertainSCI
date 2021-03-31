# Built-in forward models

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
<link rel="stylesheet" href="_static/css/main.css">

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

Authors:  


## Overview

UncertainSCI is distributed with simple forward models included. These forward models stem from a variety of applications in uncertainty quantification (UQ), and many are typical test problems that are used to validate UQ algorithms. These models are included with UncertainSCI so that users can test UQ procedures in a standalone way.

## Algebraic models

Several simple algebraic models are described below. These are functions of explicit form that allow validation and debugging of UQ algorithms. In particular, statistics of many of these examples are explicitly computable.

### Ishigami function

Given real parameters $(a,b) \in \mathbb{R}^2$, the Ishigami function is given by

\begin{align}
f(\mathbf{p}) &= \left( 1 + b p_3^4 \right) \sin p_1 + a \sin^2 p_2, & \mathbf{p} &= (p_1, p_2, p_3)
\end{align}

Because this function has an explicit ANOVA decomposition, its partial variances are explicitly computable. If each parameters $p_i$ is modeled as uniform random variable over $[-\pi, \pi]$, i.e., $p_i \sim \mathcal{U}\left([-\pi, pi]\right)$, then the variances are given by,

\begin{align}
  \mathrm{Mean}[f(\mathbf{p})] &= a\pi,  \\\\
  \mathrm{Var}[f(\mathbf{p})] &= \frac{1}{2} + \frac{a^2}{8} + \pi^4 b \left( \frac{1}{5} + \frac{\pi^4 b}{18} \right), \\\\
  \mathrm{Var}\_1 [f(\mathbf{p})] &= \frac{1}{2} \left( 1 + b \frac{\pi^4}{5}\right)^2, \\\\
  \mathrm{Var}\_2 [f(\mathbf{p})] &= \frac{a^2}{8}, \\\\
  \mathrm{Var}\_{13} [f(\mathbf{p})] &= \pi^8 b^4 \left( \frac{1}{18} - \frac{1}{50}\right)
\end{align}


### Borehole function

The Borehole function is given by 

\begin{align}
 f(\mathbf{p}) = \frac{g_1(p)}{g_2(p) g_3(p)},
\end{align}
where
\begin{align}
  g_1(p) &= 2\pi p_3 (p_4 - p_6), & 
  g_2(p) &= \log(p_2/p_1), &
  g_3(p) &= 1 + \frac{2 p_7 p_3}{g_2(p) p_1^2 p_8} + \frac{p_3}{p_5}
\end{align}
where the 8 parameters collected in $\mathbf{p}$ are
\begin{align}
  p = (p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8)
    = (r_w, r,   T_u, H_u, T_l, H_l, L,   K_w),
\end{align}
with physical interpretation and distributions as given in the table below,

| Parameter | Distribution | Description |
| --------- | ------------------ | --------------- |
| $r_w$ | $\mathcal{U} \sim [0.05, 0.15]$      | Borehole radius [m] | 
| $r$   | $\mathcal{U} \sim [100, 50,000]$     | Borehole radius of influence [m] | 
| $T_u$ | $\mathcal{U} \sim [63,070, 115,600]$ | Upper acquifer transmissivity [m^2/year] | 
| $H_u$ | $\mathcal{U} \sim [990, 1,100]$      | Upper acquifer pentiometric head [m] |
| $T_l$ | $\mathcal{U} \sim [63.1, 116]$       | Lower acquifer transmissivity [m^2/year] |
| $H_l$ | $\mathcal{U} \sim [700, 820]$        | Lower acquifer pentiometric head [m] |
| $L$   | $\mathcal{U} \sim [1,120, 1,680]$    | Borehole length [m] |
| $K_w$ | $\mathcal{U} \sim [9,885, 12,045]$   | Borehole hydraulic conductivity [m/year] |

The output function $f$ models the water flow rate. [@citation-gupta1983]

### Citations
Citations in Markdown uses [Pandoc](https://pandoc.org).  

TODO.

### Snippets
Inline snippets `like this`.  Muliple lines:
```
# # Define model
N = int(1e2)  # Number of degrees of freedom of model
left = -1.
right = 1.
x = np.linspace(left, right, N)
model = sine_modulation(N=N)
```

### Links

Internal link: [Overview](#overview)

External link: <https://www.markdownguide.org>, or [Markdown](https://www.markdownguide.org)


### Referencing Sphynx
TODO

To link the UncertainSCI API generated using Sphynx, Use this syntax: [`[text](../api_docs/pce.html#polynomial-chaos-expansions)`](../api_docs/pce.html#polynomial-chaos-expansions)
