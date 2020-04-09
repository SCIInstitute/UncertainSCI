# Contains some rudimentary (physical-space) models for testing PCE
# approximations

import numpy as np

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

def laplace_ode(left=-1., right=1., N=100, f=None):
    """

    Computes the solution to the periodic ODE:

      -d/dx [ a(x,p) d/dx u(x,p) ] = f(x),

    where, for a d-dimensional parameter p, a(x,p) has the form
    
      a(x,p) = pi^2/5 + sum_{j=1}^d p_j * sin(j*pi*x) / j^2,

    which is positive for all x if all values of p lie between [-1,1].

    Uses a finite-difference discretization of the ODE.

    """

    if f is None:
        f = lambda x: np.pi**2 * np.sin(np.pi*x)

    x = np.linspace(left, right, N+1)
    h = x[1] - x[0]
    x = x[:-1] + 0.5*h
    fx = f(x)

    def a_eval(x, p):
        a_eval = np.ones(x.shape)*np.pi**2/5
        for q in range(p.size):
            a_eval += p[q] * np.sin((q+1)*np.pi*x)/(q+1)**2
        return a_eval

    def create_system(p):
        a = a_eval(x, p)
        A = np.diag(a[1:], k=1) + np.diag(a[1:], k=-1) - (np.diag(np.roll(a, 1) + a))
        A[-1,0] = a[0]
        A[0,-1] = a[0]
        return A

    def solve_system(p):
        return np.linalg.solve(create_system(p), fx*(h**2))

    return lambda p: solve_system(p)


if __name__ == "__main__":

    import pdb
    model = laplace_ode(N=10)

    p = np.random.random(2)

    u = model(p)
