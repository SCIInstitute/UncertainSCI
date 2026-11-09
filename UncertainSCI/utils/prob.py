import numpy as np


def discrete_sampling(N, probs, states):
    """Samples iid from a discrete probability measure.

    Args:
        N: Number of iid samples generated from a random variable X.
        probs: X's probability mass function, prob(X = states[j]) = prob[j].
        states: Values of random variable X, X = states[j].

    Returns:
        N samples iid from a discrete probability measure.
    """
    p = probs[:] / sum(probs[:])

    j = np.digitize(np.random.random(N), np.cumsum(p), right=False)

    assert probs.size == states.size

    x = states[j]

    return x


# TODO: Use numpy.choice or scipy.stats.discrete_rv.


def choice(a, size=1, p=None):
    if p is None:
        p = np.ones(len(a))/len(a)
    return discrete_sampling(size, p, a)


if __name__ == "__main__":
    from families import JacobiPolynomials

    alph = -0.8
    bet = np.sqrt(101)
    states = np.linspace(-1, 1, 10)
    n = 4
    J = JacobiPolynomials(alpha=alph, beta=bet)

    probs = J.idist(states, n, M=10)

    N = 5
    print(discrete_sampling(N, probs, states))
