# Gaussian Processes

For the purposes here, we define a Gaussian process as follows:

Let $\Omega$ some domain, $k: \Omega \times \Omega \rightarrow \mathbb{R}$ a
positive-definite kernel, and $\mu: \Omega \rightarrow \mathbb{R}$ any function.
Then a random function $f: \Omega \rightarrow \mathbb{R}$ is said to be a
*Gaussian process* (Kanagawa et al. 2018) if for any set
$X = (x_1, \dots, x_n) \subset \Omega$ for any $n \in \mathbb{N}$

$$
f_X \sim \mathcal{N}^n(\mu_X, k_{XX})
$$

where $\mathcal{N}^n$ is the $n$-multivariate normal distribution, and

$$
\begin{aligned}
    f_X &= (f(x_1), \dots, f(x_n))^\top \in \mathbb{R}^n, \\
    \mu_X &= (\mu(x_1), \dots, \mu(x_n))^\top \in \mathbb{R}^n, \text{ and} \\
    (k_{XX})_{ij} &= k(x_i, x_j).
\end{aligned}
$$

Notationally, we write

$$
f \sim \mathcal{GP}(\mu, k).
$$

A *positive-definite kernel* $k$ is any symmetric function such that for any
$X = (x_1, \dots, x_n) \subset \Omega$ then for all $u \in \mathbb{R}^n$

$$
u^\top k_{XX} \,u \geq 0.
$$

Note that this is closely analogous to the standard definition of a
positive-(semi-)definite operator.

## Mathematical Notes on Implementation

### Efficient Solves of GPs with Kronecker-structure Kernels

Let a Gaussian process $\mathcal{GP}(\mu, k)$ such that $f \sim \mathcal{GP}$ where
$f: X \rightarrow Y$ where $X = \mathbb{R}^d$ and $Y = \mathbb{R}^c$. In such cases,
$k$ is a matrix-valued kernel, i.e., $k(x_i, x_j) = M \in \mathbb{R}^{c \times c}$
positive-definite for $x_i, x_j \in X$.

In some cases, it is appropriate to model $k$ with Kronecker structure,

$$
k(x_i, x_j) = k^0(x_i, x_j) \otimes C
$$

for a scalar kernel $k^0$, where $x_i, x_j \in X$, and output covariance
$C \in \mathbb{R}^{c \times c}$ positive-definite.

For notational convenience, let

$$
K = K^0 \otimes C
$$

where $K_{ij} = k(x_i, x_j)$ and $K^0_{ij} = k^0(x_i, x_j)$ for $x_i, x_j \in X$.

Let the noisy observation matrix $M = K + \operatorname{diag}(s)$.  When computing the
posterior of a Gaussian process with this structure, it is necessary to compute

$$
\begin{aligned}
    M^{-1} &= (K + \operatorname{diag}(s))^{-1} \\
           &= (K^0 \otimes C + \operatorname{diag}(s))^{-1}.
\end{aligned}
$$

Depending on the structure of $s$, this computation can be factored to accelerate
solves involving this inverse.

1. $s = \sigma^2 \mathbb{1}$ (totally isotropic noise)

    Here

    $$
    M = K^0 \otimes C + \sigma^2 I.
    $$

    Note that $K^0$ and $C$ are positive definite, thus they admit diagonlizations

    $$
    K^0 = Q \Lambda Q^\top \qquad \text{and} \qquad C = U \Gamma U^\top.
    $$

    Thus

    $$
    \begin{aligned}
        M^{-1} &= (Q \Lambda Q^\top \otimes U \Gamma U^\top + \sigma^2 I)^{-1} \\
               &= \left( (Q \otimes U)(\Lambda \otimes \Gamma + \sigma^2 I)(Q \otimes U)^\top \right)^{-1}
    \end{aligned}
    $$

    where the second line follows from $\sigma^2 I$ diagonal.
