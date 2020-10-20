import numpy as np

from UncertainSCI.families import jacobi_recurrence_values, jacobi_weight_normalized

from UncertainSCI.opoly1d import linear_modification, quadratic_modification
from UncertainSCI.opoly1d import gauss_quadrature_driver

import pdb

def gq_modification(integrand, a, b, N, roots=np.zeros(0), quadroots=np.zeros(0), Nmax=100, gamma=(0,0), leading_coefficient=1.):
    """
    Uses Gaussian quadrature to approximate

      \int_a^b q(x) integrand(x) dx,

    where integrand is an input function, and q is a polynomial defined as,

      q(x) = leading_coefficient * prod_{j=1}^R (x - roots[j-1]) * prod_{q=1}^Q (x - quadroots[j-1])**2,

    where R = roots.size and Q = quadroots.size, and it is assumed that no
    entries of roots lie inside the interval (a,b).  The coefficient
    leading_coefficient can be negative.

    The procedure rewrites the integral using the input gamma as

      \int_a^b (integrand(x) / w(x)) q(x) w(x) dx,

    where w is a mapped Jacobi weight function:

        w(x) = v(r(x)),    gamma = (alpha, beta)

    where r(x) affinely maps (a,b) to (-1,1), and v(r) is the (alpha,beta)
    Jacobi weight function in UncertainSCI.families.jacobi_weight_normalized.

    The procedure then performs measure modifications on w, absorbing q into
    the measure. An N-point Gaussian quadrature rule with respect to this
    modified measure is used to integrate (integrand / w).
    """

    assert (a < b) and (N > 0)

    # If the gamma parameters are > 0, then we write
    #   gamma[0] = gam + G,
    # where gam \in (-1,0) and G is a positive integer. And similarly for
    # gamma[1].
    # 
    # Then we take the Jacobi weight to be w associated with gam, 
    # and set N, Nmax += G
    for ind in range(2):
        if gamma[ind] > 0:
            G = np.ceil(gamma[ind])
            gam = gamma[ind] - G
            N     += int(G)
            Nmax  += int(G)
            gamma[ind] = gam

    assert (gamma[0] <=0.) and (gamma[1] <= 0.)
    Nmax = max(Nmax, N)

    R = roots.size
    Q = quadroots.size

    # Map everything to [-1,1], and scale by Jacobian afterward
    jac = 2/(b-a)
    map_to_standard = lambda x: jac*(x - a) - 1 # Maps [a,b] to [-1,1]
    map_to_ab = lambda x: (x+1)/jac + a         # Maps [-1,1], to [a,b]

    # Recurrence coefficients for appropriate Jacobi probability measure
    ab = jacobi_recurrence_values(Nmax+R+2*Q, gamma[0], gamma[1])
    ab[0,1] = 1.

    # The sign of q is determined by how many zeros lie to the right of (a,b)
    sgn = (-1)**(np.sum(roots >= b) % 2) if R > 0 else 1.

    # Modify ab by the zeros of q
    # Simultaneously gradually scale by leading_coefficient and jacobians
    sgn *= np.sign(leading_coefficient)
    if (R+2*Q) > 0:
        # Scale each linear modification by C = leading_coefficient**(1/(R+2Q))
        # There is also a jac factor since (x - x0) = 1/jac * (r-r0)
        # Finally, we need the square root of this because ab[0,1] scales like
        # the square root of an integral.
        C = np.exp(np.log(np.abs(leading_coefficient))/(R+2*Q))/jac
        Csqrt = np.sqrt(C)

        for x0 in quadroots:
            ab = quadratic_modification(ab, map_to_standard(x0))
            ab[0,1] *= C
        for x0 in roots:
            ab,_ = linear_modification(ab, map_to_standard(x0))
            ab[0,1] *= Csqrt
    else:
        ab[0,1] *= np.abs(leading_coefficient)

    x,w = gauss_quadrature_driver(ab, N)

    # Gauss quadrature
    integral = np.sum(w * integrand(map_to_ab(x))/jacobi_weight_normalized(x, gamma[0], gamma[1]))

    # Jacobian factor for the affine map and sign:
    integral *= sgn/jac

    return integral

def gq_modification_adaptive(integrand, a, b, N, N_step = 10, tol = 1e-8, **kwargs):
    s = gq_modification(integrand, a, b, N, **kwargs)
    s_new = gq_modification(integrand, a, b, N = N + N_step, **kwargs)
    
    while np.abs(s - s_new) > tol:
        s = s_new
        N += N_step
        s_new = gq_modification(integrand, a, b, N = N, **kwargs)
    return s_new

def gq_modification_composite(integrand, a, b, N, subintervals=np.zeros([0,4]), adaptive=True, **kwargs):
    """
    Uses a composite quadrature rule where each subinterval uses
    gq_modification to integrate. The integral is split into
    subintervals.shape[0] integrals. 

    subintervals is an M x 4 array, where each row has the form

      [left, right, left_sing, right_sing]

    with each subinterval being [left, right], with the appropriate left- and
    right-hand singularity strengths.

    If subintervals is empty, performs integration over an interval specified by

      [a, b, 0, 0]

    Typical keyword arguments that should be input to this function are:

        - quadroots
        - roots
        - leading_coefficient

    See gq_modification for a description of these inputs.
    """
    integral = 0.

    if subintervals.shape[0] == 0:
        subintervals = np.zeros([1, 4])
        subintervals[0,:] = [a, b, 0, 0]

    for q in range(subintervals.shape[0]):
        gamma = [subintervals[q,3], subintervals[q,2]]
        if adaptive:
            integral += gq_modification_adaptive(integrand, subintervals[q,0], subintervals[q,1], N, gamma=gamma, **kwargs)
        else:
            integral += gq_modification(integrand, subintervals[q,0], subintervals[q,1], N, gamma=gamma, **kwargs)

    return integral
