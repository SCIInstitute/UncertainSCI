# Contains some rudimentary (physical-space) models for testing PCE
# approximations. All these functions support the syntax output = f(p), where p
# is a d-dimensional vector, and output is a vector whose size is the dimension
# of the model output.

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import scipy.optimize

def ishigami_function(a, b):
    """
    Returns the Ishigami function,

    .. math::

      f(p) = (1 + b*p_3^4) \\sin p_1 + a \\sin^2 p_2,

    where :math:`p = (p_1, p_2, p_3)` are random parameters that are all
    uniformly distributed over :math:`[-\\pi, \\pi]`: :math:`p_i \\sim
    \\mathcal{U}([-\\pi, \\pi])`.
    """

    def f(p):
        return np.sin(p[0]) * (1 + b*p[2]**4) + a*np.sin(p[1])**2

    return f

def borehole_function():
    """
    Returns the Borehole function,

    .. math::

      f(p) = g_1(p) / (g_2(p) * g_3(p)),
      g_1(p) = 2\\pi p_3 * (p_4 - p_6)
      g_2(p) = log(p_2/p_1)
      g_3(p) = 1 + 2*p_7*p_3/(g_2(p) * p_1^2 * p_8) + p_3/p_5

    where the 8-dimensional parameters :math:`p` are

      p = (p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8)
        = (r_w, r,   T_u, H_u, T_l, H_l, L,   K_w)
    """

    def g1(p):
        return 2*np.pi*p[2]*(p[3] - p[5])

    def g2(p):
        return np.log(p[1]/p[0])

    def g3(p, g2val):
        return 1 + 2*p[6]*p[2]/(g2val * p[0]**2 * p[7]) + p[2]/p[4]

    def f(p):
        g2val = g2(p)
        return g1(p)/(g2val * g3(p,g2val))

    return f

def taylor_frequency(p):
    """
    Returns ( \\sum_{j=1}^d p_j^j )
    """

    return np.sum(p**(1 + np.arange(p.size)))


def sine_modulation(left=-1, right=1, N=100):
    """
    For a d-dimensional parameter p, defines the model,

      f(x,p) = sin [ pi * ( \\sum_{j=1}^d p_j^j ) * x ],

    where x is N equispaced points on the interval [left, right].

    Returns a function pointer with the syntax p ----> f(p).
    """

    x = np.linspace(left, right, N)

    return lambda p: np.sin(np.pi * x * taylor_frequency(p))


def mercer_eigenvalues_exponential_kernel(N, a, b):
    """
    For a 1D exponential covariance kernel,

      K(s,t) = exp(-|t-s| / a),     s, t \\in [-b,b],

    computes the first N eigenvalues of the associated Mercer integral
    operator.

    Precisely, computes the first N/2 positive solutions to both of the following
    transcendental equations for w and v:

       1 - a v tan(v b) = 0
       a w + tan(w b) = 0

    The eigenvalues are subsequently defined through these solutions.

    Returns (1) the N eigenvalues lamb, (2) the first ceil(N/2) solutions for
    v, (3) the first floor(N/2) solutions for w.
    """

    assert N > 0 and a > 0 and b > 0

    M = int(np.ceil(N/2))
    w = np.zeros(M)
    v = np.zeros(M)

    # First equation transformed:
    # vt = v b
    #
    #  -(b/a) / vt + tan(vt) = 0

    def f(x):
        return -(b/a)/x + np.tan(x)

    for n in range(M):
        # Compute bracketing interval
        # root somewhere in right-hand part of [2*n-1, 2*n+1]*pi/2 interval
        RH_value = -1
        k = 4
        while RH_value < 0:
            k += 1
            right = (2*n+1)*np.pi/2 - 1/k
            RH_value = f(right)

        # Root can't be on LHS of interval
        if n == 0:
            left = 1/k
            while f(left) > 0:
                k += 1
                left = 1/k
        else:
            left = n*np.pi

        v[n] = scipy.optimize.brentq(f, left, right)

    v /= b

    # Second equation transformed:
    # wt = w b
    #
    #  (a/b) wt + tan(wt) = 0

    def f(x):
        return (a/b)*x + np.tan(x)

    for n in range(M):

        # Compute bracketing interval
        # root somewhere in [2*n+1, 2*n+3]*pi/2
        LH_value = 1
        k = 4
        while LH_value > 0:
            k += 1
            left = (2*n+1)*np.pi/2 + 1/k
            LH_value = f(left)

        # Root can't be on RHS of interval
        right = (n+1)*np.pi

        w[n] = scipy.optimize.brentq(f, left, right)

    w /= b

    if (N % 2) == 1:  # Don't need last root for w
        w = w[:-1]

    lamb = np.zeros(N)
    oddinds = [i for i in range(N) if (i % 2) == 0]  # Well, odd for 1-based indexing
    lamb[oddinds] = 2*a/(1+(a*v)**2)
    eveninds = [i for i in range(N) if (i % 2) == 1]  # even for 1-based indexing
    lamb[eveninds] = 2*a/(1+(a*w)**2)

    return lamb, v, w


def KLE_exponential_covariance_1d(N, a, b, mn):
    """
    Returns a pointer to a function the evaluates an N-term Karhunen-Loeve
    Expansion of a stochastic process with exponential covariance function on a
    bounded interval [-b,b]. Let the GP have the covariance function,

        C(s,t) = exp(-|t-s| / a),

    and mean function given by mn. Then the N-term KLE of the process is given
    by

        K_N(x,P) = mn(x) + \\sum_{n=1}^N P_n sqrt(\\lambda_n) \\phi_n(x),

    where (lambda_n, phi_n) are the leading eigenpairs of the associated Mercer
    kernel. The eigenvalues are computed in
    mercer_eigenvalues_exponential_kernel. The (P_n) are iid standard normal
    Gaussian random variables.

    Returns a function lamb(x,P) that takes in a 1D np.ndarray x and a 1D
    np.ndarray vector P and returns the KLE realization on x for that value of
    P.
    """

    lamb, v, w = mercer_eigenvalues_exponential_kernel(N, a, b)

    efuns = N*[None]
    for i in range(N):
        if (i % 2) == 0:
            i2 = int(i/2)
            efuns[i] = (lambda i2: lambda x: np.cos(v[i2]*x) / np.sqrt(b + np.sin(2*v[i2]*b)/(2*v[i2])))(i2)
        else:
            i2 = int((i-1)/2)
            efuns[i] = (lambda i2: lambda x: np.sin(w[i2]*x) / np.sqrt(b - np.sin(2*w[i2]*b)/(2*w[i2])))(i2)

    def KLE(x, p):
        return mn(x) + np.array([np.sqrt(lamb[i])*efuns[i](x) for i in range(N)]).T @ p

    return KLE


def laplace_ode_diffusion(x, p):
    """ Parameterized diffusion coefficient for 1D ODE

    For a d-dimensional parameter p, the diffusion coefficient a(x,p) has the form

      a(x,p) = pi^2/5 + sum_{j=1}^d p_j * sin(j*pi*(x+1)/2) / j^2,

    which is positive for all x if all values of p lie between [-1,1].
    """

    a_val = np.ones(x.shape)*np.pi**2/5
    for q in range(p.size):
        a_val += p[q] * np.sin((q+1)*np.pi*(x+1)/2)/(q+1)**2
    return a_val


def laplace_grid_x(left, right, N):
    """
    Computes one-dimensional equispaced grid with N points on the interval
    (left, right).
    """
    return np.linspace(left, right, N)


def laplace_ode(left=-1., right=1., N=100, f=None, diffusion=laplace_ode_diffusion):
    """

    Computes the solution to the ODE:

      -d/dx [ a(x,p) d/dx u(x,p) ] = f(x),

    with homogeneous Dirichlet boundary conditions at x = left, x = right.

    For a d-dimensional parameter p, a(x,p) is the function defined in laplace_ode_diffusion.

    Uses an equispaced finite-difference discretization of the ODE.

    """

    assert N > 2

    if f is None:
        def f(x):
            return np.pi**2 * np.cos(np.pi*x)

    x = laplace_grid_x(left, right, N)
    h = x[1] - x[0]
    fx = f(x)

    # Set homogeneous Dirichlet conditions
    fx[0], fx[-1] = 0., 0.

    # i+1/2 points
    xh = x[:-1] + h/2.

    def create_system(p):
        nonlocal x, xh, N
        a = diffusion(xh, p)
        number_nonzeros = 1 + 1 + (N-2)*3
        rows = np.zeros(number_nonzeros, dtype=int)
        cols = np.zeros(number_nonzeros, dtype=int)
        vals = np.zeros(number_nonzeros, dtype=float)

        # Set the homogeneous Dirichlet conditions
        rows[0], cols[0], vals[0] = 0, 0, 1.
        rows[1], cols[1], vals[1] = N-1, N-1, 1.
        ind = 2

        for q in range(1, N-1):
            # Column q-1
            rows[ind], cols[ind], vals[ind] = q, q-1, -a[q-1]
            ind += 1

            # Column q
            rows[ind], cols[ind], vals[ind] = q, q, a[q-1] + a[q]
            ind += 1

            # Column q+1
            rows[ind], cols[ind], vals[ind] = q, q+1, -a[q]
            ind += 1

        A = sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))

        return A

    def solve_system(p):
        nonlocal fx, h
        return splinalg.spsolve(create_system(p), fx*(h**2))

    return lambda p: solve_system(p)

def laplace_ode_1d(Nparams, a=1., b=1., abar=3., N=100):
    """ Generates a 1D Laplace ODE model with parameterized diffusion.

    Define model:

    -d/dx a(x,p) d/dx u(x,p) = f(x)

    over x in [-1,1], where a(x,p) is a parameterized diffusion model:

    a(x,p) = abar(x) + sum_{j=1}^d lambda_j Y_j phi_j(x),

    where d = Nparams, (lambda_j, phi_j) are eigenpairs of the exponential
    covariance kernel,

      K(s,t) = exp(-|s-t|/a).

    The Y_j are modeled as iid random variables.
    """

    abarfun = lambda x: abar*np.ones(np.shape(x))
    KLE = KLE_exponential_covariance_1d(Nparams, a, b, abarfun)

    diffusion = lambda x, p: KLE(x, p)
    x = laplace_grid_x(-b, b, N)

    return x, laplace_ode(left=-b, right=b, N=N, diffusion=diffusion)

def laplace_grid_xy(left, right, N1, down, up, N2):
    """
    Computes two-dimensional tensorial equispaced grid corresponding to the
    tensorization of N1 equispaced points on the interval (left, right) and N2
    equispaced points on the interval (down, up).
    """
    x = np.linspace(left, right, N1)
    y = np.linspace(down, up, N2)

    X, Y = np.meshgrid(x, y)

    return X.flatten(order='C'), Y.flatten(order='C')


def laplace_pde_diffusion(x, p):
    """ Parameterized diffusion coefficient for 2D PDE

    For a d-dimensional parameter p, the diffusion coefficient a(x,p) has the form

      a(x,p) = pi^2/5 + sum_{j=1}^d p_j * sin(j*pi*(x+1)/2) / j^2,

    which is positive for all x if all values of p lie between [-1,1].
    """

    a_val = np.ones(x.shape)*np.pi**2/5
    for q in range(p.size):
        a_val += p[q] * np.sin((q+1)*np.pi*(x+1)/2)/(q+1)**2
    return a_val


def genz_oscillatory(w=0., c=None):
    """
    Returns a pointer to the "oscillatory" Genz test function defined as

       f(p) = \\cos{ 2\\pi w + \\sum_{i=1}^dim c_i p_i }

    where p \\in R^d. The default value for w is 0, and that for c is a
    d-dimensional vector of ones.
    """

    def cos_eval(p):
        nonlocal c
        if c is None:
            c = np.ones(p.size)
        return np.cos(2*np.pi*w + np.dot(c, p))

    return lambda p: cos_eval(p)


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import scipy as sp

    dim = 5

    a = 3
    b = 1

    def mn(x):
        return np.zeros(x.shape)

    KLE = KLE_exponential_covariance_1d(dim, a, b, mn)

    def diffusion(x, p):
        return np.exp(KLE(x, p))

    left = -1.
    right = 1.
    N = 1000
    model = laplace_ode(left=left, right=right, N=N, diffusion=diffusion)
    x = laplace_grid_x(left, right, N)

    K = 4
    p = K*[None]
    u = K*[None]
    a = K*[None]

    for k in range(K):
        p[k] = np.random.rand(dim)*2 - 1
        # a[k] = laplace_ode_diffusion(x, p[k])
        a[k] = diffusion(x, p[k])
        u[k] = model(p[k])

    for k in range(K):

        row = np.floor(k/2) + 1
        col = k % (K/2) + 1

        index = col + (row-1)*K/2

        plt.subplot(2, K, k+1)
        plt.plot(x, a[k], 'r')
        plt.title('Diffusion coefficient')
        plt.ylim([0, 3.0])

        plt.subplot(2, K, k+1+K)
        plt.plot(x, u[k])
        plt.title('Solution u')
        plt.ylim([-5, 5])

    M = 1000
    U = np.zeros([u[0].size, M])
    for m in range(M):
        U[m, :] = model(np.random.rand(dim)*2 - 1)

    _, svs, _ = np.linalg.svd(U)
    _, r, _ = sp.linalg.qr(U, pivoting=True)

    plt.figure()
    plt.semilogy(svs[:100], 'r')
    plt.semilogy(np.abs(np.diag(r)[:100]), 'b')
    plt.legend(["Singular values", "Orthogonalization residuals"])
    plt.show()
