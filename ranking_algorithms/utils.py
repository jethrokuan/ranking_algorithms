"""Utilities for various"""

import warnings

import numpy as np
from scipy import linalg


def exp_transform(params):
    """Transform parameters into exp-scale weights."""
    weights = np.exp(np.asarray(params) - np.mean(params))
    return (len(weights) / weights.sum()) * weights


def log_transform(weights):
    """Transform weights into centered log-scale parameters."""
    params = np.log(weights)
    return params - params.mean()


def statdist(generator):
    """Compute the stationary distribution of a Markov chain.
    Parameters
    ----------
    generator : array_like
        Infinitesimal generator matrix of the Markov chain.
    Returns
    -------
    dist : numpy.ndarray
        The unnormalized stationary distribution of the Markov chain.
    Raises
    ------
    ValueError
        If the Markov chain does not have a unique stationary distribution.
    """
    generator = np.asarray(generator)
    n = generator.shape[0]
    with warnings.catch_warnings():
        # The LU decomposition raises a warning when the generator matrix is
        # singular (which it, by construction, is!).
        warnings.filterwarnings("ignore")
        lu, _ = linalg.lu_factor(generator.T, check_finite=False)
    # The last row contains 0's only.
    left = lu[:-1, :-1]
    right = -lu[:-1, -1]
    # Solves system `left * x = right`. Assumes that `left` is
    # upper-triangular (ignores lower triangle).
    try:
        res = linalg.solve_triangular(left, right, check_finite=False)
    except:
        # Ideally we would like to catch `linalg.LinAlgError` only,
        # but there seems to be a bug in scipy, in the code that
        # raises the LinAlgError (!!).
        raise ValueError(
            "stationary distribution could not be computed. "
            "Perhaps the Markov chain has more than one absorbing class?")
    res = np.append(res, 1.0)
    return (n / res.sum()) * res
