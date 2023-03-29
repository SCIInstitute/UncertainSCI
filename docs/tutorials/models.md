# Built-in forward models

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
<link rel="stylesheet" href="_static/css/main.css">

This project was supported by grants from the National Institute of Biomedical Imaging and Bioengineering (U24EB029012) from the National Institutes of Health.

## Overview

UncertainSCI is distributed with simple forward models included. These forward models stem from a variety of applications in uncertainty quantification (UQ), and many are typical test problems that are used to validate UQ algorithms. These models are included with UncertainSCI so that users can test UQ procedures in a standalone way.

## Algebraic models

Several simple algebraic models are described below. These are functions of explicit form that allow validation and debugging of UQ algorithms. In particular, statistics of many of these examples are explicitly computable.

### Ishigami function

Given real parameters \\((a,b) \in \mathbb{R}^2\\), the Ishigami function is given by

\begin{align}
f(\mathbf{p}) &= \left( 1 + b p_3^4 \right) \sin p_1 + a \sin^2 p_2, & \mathbf{p} &= (p_1, p_2, p_3)
\end{align}

Because this function has an explicit ANOVA decomposition, its partial variances are explicitly computable. If each parameters \\(p_i\\) is modeled as uniform random variable over \\([-\pi, \pi]\\), i.e., \\(p_i \sim \mathcal{U}\left([-\pi, \pi]\right)\\), then the variances are given by,

\begin{align}
  \mathrm{Mean}[f(\mathbf{p})] &= \frac{a}{2},  \\\\
  \mathrm{Var}[f(\mathbf{p})] &= \frac{1}{2} + \frac{a^2}{8} + \pi^4 b \left( \frac{1}{5} + \frac{\pi^4 b}{18} \right), \\\\
  \mathrm{Var}\_1 [f(\mathbf{p})] &= \frac{1}{2} \left( 1 + b \frac{\pi^4}{5}\right)^2, \\\\
  \mathrm{Var}\_2 [f(\mathbf{p})] &= \frac{a^2}{8}, \\\\
  \mathrm{Var}\_{13} [f(\mathbf{p})] &= \pi^8 b^4 \left( \frac{1}{18} - \frac{1}{50}\right)
\end{align}


### Borehole function

The Borehole function is given by 

\begin{align}
 f(\mathbf{p}) = \frac{g_1(\mathbf{p})}{g_2(\mathbf{p}) g_3(\mathbf{p})},
\end{align}
where
\begin{align}
  g_1(\mathbf{p}) &= 2\pi p_3 (p_4 - p_6), & 
  g_2(\mathbf{p}) &= \log(p_2/p_1), &
  g_3(\mathbf{p}) &= 1 + \frac{2 p_7 p_3}{g_2(\mathbf{p}) p_1^2 p_8} + \frac{p_3}{p_5}
\end{align}
where the 8 parameters collected in \\(\mathbf{p}\\) are
\begin{align}
  \mathbf{p} = (p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8)
             = (r_w, r,   T_u, H_u, T_l, H_l, L,   K_w),
\end{align}
with physical interpretation and typical distributions as given below,

- \\(r_w \sim \mathcal{U}([0.05, 0.15])\\):  Borehole radius [m]
- \\(r   \sim \mathcal{U}([100, 50,000])\\): Borehole radius of influence [m]
- \\(T_u \sim \mathcal{U}([63,070, 115,600])\\): Upper acquifer transmissivity [m^2/year]
- \\(H_u \sim \mathcal{U}([990, 1,100])\\): Upper acquifer pentiometric head [m]
- \\(T_l \sim \mathcal{U}([63.1, 116])\\): Lower acquifer transmissivity [m^2/year]
- \\(H_l \sim \mathcal{U}([700, 820])\\): Lower acquifer pentiometric head [m]
- \\(L   \sim \mathcal{U}([1,120, 1,680])\\): Borehole length [m]
- \\(K_w \sim \mathcal{U}([9,885, 12,045])\\): Borehole hydraulic conductivity [m/year]

The output function \\(f\\) models the water flow rate. [@citation-gupta1983]

## Differential equation models

More complicated forward models are included in this section, which are solutions to parametric differential equations.

### ODE boundary value problem

This model in this example is the solution to an ordinary differential equation (ODE) with random parameters $\mathbf{p}$. The solution \\(u = u(x,\mathbf{p})\\) depends on the spatial variable \\(x\\) lying on the compact domain \\([x_-, x_+]\\), and is governed by the ODE and boundary conditions,

\begin{align}
  -\frac{d}{dx} \left( a(x,\mathbf{p}) \frac{d}{dx} u(x,\mathbf{p}) \right) &= f(x), & x &\in (x_-, x_+) \\\\
  u(x_-, \mathbf{p}) &= 0, & u(x_+, \mathbf{p}) &= 0.
\end{align}
Above, the forcing function \\(f\\) and the diffusion coefficient \\(a(x, \mathbf{p})\\) are specified functions. The diffusion coefficient must be strictly positive for every parameter value $\mathbf{p}$ to ensure the above ODE is well posed. One particular specification of the diffusion coefficient is as a (truncated) Karhunen-Loeve expansion (KLE),
\begin{align}
 a(x,\mathbf{p}) &= a_0(x) + \sum_{j=1}^d p_j a_j(x), & \mathbf{p} &= (p_1, \ldots, p_d),
\end{align}
where the mean diffusion behavior \\(a_0\\) is given, and the functions \\(a_j\\) for \\(j = 1, \ldots, d\\) are (scaled) eigenfunctions of the integral operator associated to an assumed covariance kernel of a random field \\(a(x)\\). In UncertainSCI, one example of such a truncated KLE associated to an exponential covariance kernel can be created using the following:
```
imoprt numpy as np
from UncertainSCI.model_examples import KLE_exponential_covariance_1d

xminus = -1.
xplus = 1.
abar = lambda x: 3*np.ones(np.shape(x))  # Constant a_0 equal to 3
d = 4

a = KLE_exponential_covariance_1d(d, xminus, xplus, abar)
```

The output `a` is a function with two inputs \\(x\\) and \\(\mathbf{p}\\). We discretize the \\(x\\) variable with \\(N\\) equispaced points on \\([x_-, x_+]\\) and use a finite-difference method to solve the ODE. The following code then creates the ODE model with this parametric diffusion coefficient:

```
from UncertainSCI.model_examples import laplace_grid_x, laplace_ode

N = 100
x = laplace_grid_x(xminus, xplus, N)

f = lambda x: np.pi**2 * np.cos(np.pi*x)

u = laplace_ode(left=xminus, right=xplus, N=N, diffusion=a, f=f)
```

This model can be queried using the syntax `u(p)`, where `p` is a `d`-dimensional parameter vector.

# Using UncertainSCI with External Simulation Software

UncertainSCI's non-invasive methods and architecture allow it to be used with simulations run with a variety of software.  The only requirements is to take parameter sets from UncertainSCI and to generate a set of solutions for UncertainSCI to use.  This can be acheived with software that is implemented in Python or contains a Python API, or by creating a hard disk data passing system.  We will include some examples on how to use these systems to integrate simulations with UncertainSCI.   

We most often interface UncertainSCI with [SCIRun](https://github.com/SCIInstitute/SCIRun), a simulation software we also produce, to UQ predictions on Bioelectric field simulations.  

## SCIRun/UncertainSCI ECG uncertainty due to cardiac postion

An uncertainty quantification (UQ) example computing the effect of heart position on boundary element method (BEM) ECG forward computations.  This example is similar to the work of [Swenson, etal.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3362042/) but is implemented in UncertainSCI and SCIRun.  

Code and example data are found on GitHub: <https://github.com/SCIInstitute/UQExampleBEMHeartPosition>
