# Contains some rudimentary (physical-space) models for testing PCE
# approximations. All these functions support the syntax output = f(p), where p
# is a d-dimensional vector, and output is a vector whose size is the dimension
# of the model output.

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

def taylor_frequency(p):
    """
    Returns ( \sum_{j=1}^d p_j^j )
    """

    return np.sum(p**(1 + np.arange(p.size)))

def sine_modulation(left=-1, right=1, N=100):
    """
    For a d-dimensional parameter p, defines the model,

      f(x,p) = sin [ pi * ( \sum_{j=1}^d p_j^j ) * x ],

    where x is N equispaced points on the interval [left, right].

    Returns a function pointer with the syntax p ----> f(p).
    """

    x = np.linspace(left, right, N)

    return lambda p: np.sin(np.pi * x * taylor_frequency(p))

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

def laplace_ode(left=-1., right=1., N=100, f=None):
    """

    Computes the solution to the ODE:

      -d/dx [ a(x,p) d/dx u(x,p) ] = f(x),

    with homogeneous Dirichlet boundary conditions at x = left, x = right.

    For a d-dimensional parameter p, a(x,p) is the function defined in laplace_ode_diffusion.
    
    Uses an equispaced finite-difference discretization of the ODE.

    """

    assert N > 2

    if f is None:
        f = lambda x: np.pi**2 * np.cos(np.pi*x)

    x = laplace_grid_x(left, right, N)
    h = x[1] - x[0]
    fx = f(x)

    # Set homogeneous Dirichlet conditions
    fx[0], fx[-1] = 0., 0.

    # i+1/2 points
    xh = x[:-1] + h/2.

    def create_system(p):
        nonlocal x, xh, N
        a = laplace_ode_diffusion(xh, p)
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

        A = sparse.csc_matrix((vals, (rows, cols)), shape=(N,N))

        return A

    def solve_system(p):
        nonlocal fx, h
        return splinalg.spsolve(create_system(p), fx*(h**2))

    return lambda p: solve_system(p)

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

       f(p) = \cos{ 2\pi w + \sum_{i=1}^dim c_i p_i }

    where p \in R^d. The default value for w is 0, and that for c is a
    d-dimensional vector of ones.
    """

    def cos_eval(p):
        nonlocal c
        if c is None:
            c = np.ones(p.size)
        return np.cos(2*np.pi*w + np.dot(c,p))

    return lambda p: cos_eval(p)

if __name__ == "__main__":

    import pdb
    from matplotlib import pyplot as plt

    dim = 5

    left = -1.
    right = 1.
    N = 100000
    model = laplace_ode(left=left, right=right, N=N)

    p = np.random.rand(5)*2 - 1
    u = model(p)
    x = laplace_grid_x(left, right, N)
    a = laplace_ode_diffusion(x, p)

    plt.subplot(121)
    plt.plot(x,a)
    plt.title('Diffusion coefficient')

    plt.subplot(122)
    plt.plot(x,u)
    plt.title('Solution u')

