import numpy as np

def compute_subintervals(a, b, singularity_list):
    """
    Returns an M x 4 numpy array, where each row contains the left-hand point,
    right-hand point, left-singularity strength, and right-singularity
    strength.
    """

    # Tolerance for resolving internal versus boundary singularities.
    tol = 1e-12
    singularities  = np.array([entry[0] for entry in singularity_list])
    strength_left  = np.array([entry[1] for entry in singularity_list])
    strength_right = np.array([entry[2] for entry in singularity_list])

    # We can discard any singularities that lie to the left of a or the right of b
    discard = []
    for (ind,s) in enumerate(singularities):
        if s < a-tol or s > b+tol:
            discard.append(ind)

    singularities  = np.delete(singularities, discard)
    strength_left  = np.delete(strength_left, discard)
    strength_right = np.delete(strength_right, discard)

    # Sort remaining valid singularities
    order = np.argsort(singularities)
    singularities  = singularities[order]
    strength_left  = strength_left[order]
    strength_right = strength_right[order]

    # Make sure there aren't doubly-specified singularities
    if np.any(np.diff(singularities) < tol):
        raise ValueError("Overlapping singularities were specified. Singularities must be unique")

    S = singularities.size

    if S > 0:

        # Extend the singularities lists if we need to add interval endpoints
        a_sing = np.abs(singularities[0] - a) <= tol
        b_sing = np.abs(singularities[-1] - b) <= tol

        # Figure out if singularities match endpoints
        if not b_sing:
            singularities = np.hstack([singularities, b])
            strength_left = np.hstack([strength_left, 0])
            strength_right = np.hstack([strength_right, 0]) # Doesn't matter
        if not a_sing:
            singularities = np.hstack([a, singularities])
            strength_left = np.hstack([0, strength_left])  # Doesn't matter
            strength_right = np.hstack([0, strength_right]) # Doesn't matter


        # Use the singularities lists to identify subintervals
        S = singularities.size
        subintervals = np.zeros([S-1, 4])
        for q in range(S-1):
            subintervals[q,:] = [singularities[q], singularities[q+1], strength_right[q], strength_left[q+1]]

    else:

        subintervals = np.zeros([1, 4])
        subintervals[0,:] = [a, b, 0, 0]

    return subintervals

if __name__ == '__main__':
    a = -1
    b = 1
    singularity_list = []
    print (compute_subintervals(a, b, singularity_list))
