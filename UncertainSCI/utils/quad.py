import numpy as np

from UncertainSCI.families import jacobi_recurrence_values,\
                                  jacobi_weight_normalized

from UncertainSCI.opoly1d import linear_modification, quadratic_modification
from UncertainSCI.opoly1d import gauss_quadrature_driver

# from UncertainSCI.utils.compute_subintervals import compute_subintervals


def compute_subintervals(a, b, singularity_list):
    """
    Returns an M x 4 numpy array, where each row contains the left-hand point,
    right-hand point, left-singularity strength, and right-singularity
    strength.
    """

    # Tolerance for resolving internal versus boundary singularities.
    tol = 1e-12
    singularities = np.array([entry[0] for entry in singularity_list])
    strength_left = np.array([entry[1] for entry in singularity_list])
    strength_right = np.array([entry[2] for entry in singularity_list])

    # We can discard any singularities that lie to the left of a or the right
    # of b
    discard = []
    for (ind, s) in enumerate(singularities):
        if s < a-tol or s > b+tol:
            discard.append(ind)

    singularities = np.delete(singularities, discard)
    strength_left = np.delete(strength_left, discard)
    strength_right = np.delete(strength_right, discard)

    # Sort remaining valid singularities
    order = np.argsort(singularities)
    singularities = singularities[order]
    strength_left = strength_left[order]
    strength_right = strength_right[order]

    # Make sure there aren't doubly-specified singularities
    if np.any(np.diff(singularities) < tol):
        raise ValueError("Overlapping singularities were specified. \
                          Singularities must be unique")

    S = singularities.size

    if S > 0:

        # Extend the singularities lists if we need to add interval endpoints
        a_sing = np.abs(singularities[0] - a) <= tol
        b_sing = np.abs(singularities[-1] - b) <= tol

        # Figure out if singularities match endpoints
        if not b_sing:
            singularities = np.hstack([singularities, b])
            strength_left = np.hstack([strength_left, 0])
            strength_right = np.hstack([strength_right, 0])  # Doesn't matter
        if not a_sing:
            singularities = np.hstack([a, singularities])
            strength_left = np.hstack([0, strength_left])  # Doesn't matter
            strength_right = np.hstack([0, strength_right])  # Doesn't matter

        # Use the singularities lists to identify subintervals
        S = singularities.size
        subintervals = np.zeros([S-1, 4])
        for q in range(S-1):
            subintervals[q, :] = [singularities[q], singularities[q+1],
                                  strength_right[q], strength_left[q+1]]

    else:

        subintervals = np.zeros([1, 4])
        subintervals[0, :] = [a, b, 0, 0]

    return subintervals


def gq_modification(integrand, a, b, N, roots=np.zeros(0),
                    quadroots=np.zeros(0), Nmax=100,
                    gamma=(0, 0), leading_coefficient=1.):
    """
    Uses Gaussian quadrature to approximate

      \\int_a^b q(x) integrand(x) dx,

    where integrand is an input function, and q is a polynomial defined as,

      q(x) = leading_coefficient * prod_{j=1}^R (x - roots[j-1]) *
                                   prod_{q=1}^Q (x - quadroots[j-1])**2,

    where R = roots.size and Q = quadroots.size, and it is assumed that no
    entries of roots lie inside the interval (a,b).  The coefficient
    leading_coefficient can be negative.

    The procedure rewrites the integral using the input gamma as

      \\int_a^b (integrand(x) / w(x)) q(x) w(x) dx,

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
            N += int(G)
            Nmax += int(G)
            gamma[ind] = gam

    assert (gamma[0] <= 0.) and (gamma[1] <= 0.)
    Nmax = max(Nmax, N)

    R = roots.size
    Q = quadroots.size

    # Map everything to [-1,1], and scale by Jacobian afterward
    jac = 2/(b-a)

    def map_to_standard(x):  # Maps [a,b] to [-1,1]
        return jac*(x - a) - 1
    # map_to_standard = lambda x: jac*(x - a) - 1

    def map_to_ab(x):  # Maps [-1,1], to [a,b]
        return (x + 1) / jac + a
    # map_to_ab = lambda x: (x+1)/jac + a

    # Recurrence coefficients for appropriate Jacobi probability measure
    ab = jacobi_recurrence_values(Nmax+R+2*Q, gamma[0], gamma[1])
    ab[0, 1] = 1.

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
            ab[0, 1] *= C
        for x0 in roots:
            ab = linear_modification(ab, map_to_standard(x0))
            ab[0, 1] *= Csqrt
    else:
        ab[0, 1] *= np.abs(leading_coefficient)

    x, w = gauss_quadrature_driver(ab, N)

    # Gauss quadrature
    integral = np.sum(w * integrand(map_to_ab(x)) /
                      jacobi_weight_normalized(x, gamma[0], gamma[1]))

    # Jacobian factor for the affine map and sign:
    integral *= sgn/jac

    return integral


def gq_modification_adaptive(integrand, a, b, N, N_step=10, tol=1e-12,
                             **kwargs):
    s = gq_modification(integrand, a, b, N, **kwargs)
    s_new = gq_modification(integrand, a, b, N=N + N_step, **kwargs)

    while np.abs(s - s_new) > tol:
        s = s_new
        N += N_step
        s_new = gq_modification(integrand, a, b, N=N, **kwargs)
    return s_new


def gq_modification_composite(integrand, a, b, N,
                              subintervals=np.zeros([0, 4]), adaptive=True,
                              **kwargs):
    """
    Uses a composite quadrature rule where each subinterval uses
    gq_modification to integrate. The integral is split into
    subintervals.shape[0] integrals.

    subintervals is an M x 4 array, where each row has the form

      [left, right, left_sing, right_sing]

    with each subinterval being [left, right], with the appropriate left- and
    right-hand singularity strengths.

    If subintervals is empty, performs integration over an interval specified
    by

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
        subintervals[0, :] = [a, b, 0, 0]

    for q in range(subintervals.shape[0]):
        gamma = [subintervals[q, 3], subintervals[q, 2]]
        if adaptive:
            integral += gq_modification_adaptive(integrand, subintervals[q, 0],
                                                 subintervals[q, 1], N,
                                                 gamma=gamma, **kwargs)
        else:
            integral += gq_modification(integrand, subintervals[q, 0],
                                        subintervals[q, 1], N, gamma=gamma,
                                        **kwargs)

    return integral


def gq_modification_unbounded_composite(integrand, a, b, N, singularity_list,
                                        adaptive=True, tol=1e-12, step=1,
                                        **kwargs):
    """
    Functional as the same as util.quad.gq_modification_composite but with a
    unbounded interval [a,b] I expect this method lying in the util.quad, but
    the issue is:

    1. Subintervals are changed when extending the intervals that we're
    integrating on, thus subintervals cannot be a argument, instead, We use
    singularity_list.

    2. Method compute_subintervals in the composite is required to obtain the
    subintervals for different interval [a,b]

    So We temporarily put this method here, maybe change this later for bettter
    consideration.
    """

    if a == -np.inf and b == np.inf:
        # l = -1.; r = 1.
        le, r = -1., 1.
        subintervals = compute_subintervals(le, r, singularity_list)
        integral = gq_modification_composite(integrand, le, r, N, subintervals,
                                             adaptive, **kwargs)

        integral_new = 1.
        while np.abs(integral_new) > tol:
            r = le
            le = r - step
            subintervals = compute_subintervals(le, r, singularity_list)
            integral_new = gq_modification_composite(integrand, le, r, N,
                                                     subintervals, adaptive,
                                                     **kwargs)
            integral += integral_new

        le, r = -1., 1.
        # l = -1.; r = 1.
        integral_new = 1.
        while np.abs(integral_new) > tol:
            le = r
            r = le + step
            subintervals = compute_subintervals(le, r, singularity_list)
            integral_new = gq_modification_composite(integrand, le, r, N,
                                                     subintervals, adaptive,
                                                     **kwargs)
            integral += integral_new

    elif a == -np.inf:
        r = b
        le = b - step
        subintervals = compute_subintervals(le, r, singularity_list)
        integral = gq_modification_composite(integrand, le, r, N, subintervals,
                                             adaptive, **kwargs)
        integral_new = 1.
        while np.abs(integral_new) > tol:
            r = le
            le = r - step
            subintervals = compute_subintervals(le, r, singularity_list)
            integral_new = gq_modification_composite(integrand, le, r, N,
                                                     subintervals, adaptive,
                                                     **kwargs)
            integral += integral_new

    elif b == np.inf:
        le = a
        r = a + step
        subintervals = compute_subintervals(le, r, singularity_list)
        integral = gq_modification_composite(integrand, le, r, N, subintervals,
                                             adaptive, **kwargs)
        integral_new = 1.
        while np.abs(integral_new) > tol:
            le = r
            r = le + step
            subintervals = compute_subintervals(le, r, singularity_list)
            integral_new = gq_modification_composite(integrand, le, r, N,
                                                     subintervals, adaptive,
                                                     **kwargs)
            integral += integral_new

    return integral
