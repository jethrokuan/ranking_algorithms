"""Luce Spectral Ranking.


"""

import pickle
import csv
import os
import argparse
import functools
import numpy as np
from ranking_algorithms.utils import exp_transform, log_transform, statdist
from ranking_algorithms.convergence_tests import NormOfDifferenceTest


def _init_lsr(n_items, alpha, initial_params=None):
    """Initializes the LSR markov chain and the weights.

    :param n_items: Number of items
    :param alpha:
    :param initial_params:
    :returns:
    :rtype:

    """
    if initial_params is None:
        weights = np.ones(n_items)
    else:
        weights = exp_transform(initial_params)
    chain = alpha * np.ones((n_items, n_items), dtype=float)
    return weights, chain


def _ilsr(fun, params, max_iter, tol):
    """Iteratively refines LSR estimates until convergence.

    :param fun:
    :param params:
    :param max_iter:
    :param tol:
    :returns:
    :rtype:

    """
    converged = NormOfDifferenceTest(tol, order=1)
    for _ in range(max_iter):
        params = fun(initial_params=params)
        if converged(params):
            return params
    raise RuntimeError(
        "Did not converge after {} iterations.".format(max_iter))


def lsr_pairwise(n_items, data, alpha=0., initial_params=None):
    """Computes the LSR estimate of model parameters.

    :param n_items:
    :param data:
    :param alpha:
    :param initial_params:
    :returns:
    :rtype:

    """
    weights, chain = _init_lsr(n_items, alpha, initial_params)
    for winner, loser in data:
        chain[loser, winner] += 1 / (weights[winner] + weights[loser])
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def ilsr_pairwise(n_items,
                  data,
                  alpha=0.0,
                  initial_params=None,
                  max_iter=100,
                  tol=1e-8):
    """Compute the ML estimate of model parameters using I-LSR.
    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise-comparison data (see :ref:`data-pairwise`), using
    the iterative Luce Spectral Ranking algorithm [MG15]_.
    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).
    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.
    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    fun = functools.partial(
        lsr_pairwise, n_items=n_items, data=data, alpha=alpha)
    return _ilsr(fun, initial_params, max_iter, tol)


def rank_centrality(n_items, data, alpha=0.0):
    """Compute the Rank Centrality estimate of model parameters.
    This function implements Negahban et al.'s Rank Centrality algorithm
    [NOS12]_. The algorithm is similar to :func:`~choix.ilsr_pairwise`, but
    considers the *ratio* of wins for each pair (instead of the total count).
    The transition rates of the Rank Centrality Markov chain are initialized
    with ``alpha``. When ``alpha > 0``, this corresponds to a form of
    regularization (see :ref:`regularization` for details).
    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
    alpha : float, optional
        Regularization parameter.
    Returns
    -------
    params : numpy.ndarray
        An estimate of model parameters.
    """
    _, chain = _init_lsr(n_items, alpha)
    for winner, loser in data:
        chain[loser, winner] += 1.0
    # Transform the counts into ratios.
    idx = chain > 0  # Indices (i,j) of non-zero entries.
    chain[idx] = chain[idx] / (chain + chain.T)[idx]
    # Finalize the Markov chain by adding the self-transition rate.
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def get_data(data_path):
    data = list()
    with open(data_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for winner, loser in csv_reader:
            winner = int(winner)
            loser = int(loser)
            data.append((winner, loser))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file",
                        help="path to data.",
                        required=True)

    args = parser.parse_args()
    idx2node_path = os.path.join(args.file, "idx2node.pkl")

    with open(idx2node_path, "rb") as f:
        idx2node = pickle.load(f)

    data_path = os.path.join(args.file, "data.csv")

    data = get_data(data_path)

    # scores = ilsr_pairwise(len(idx2node), data, alpha=0.2)
    scores = rank_centrality(len(idx2node), data, alpha=0.2)
    node2score = {idx2node[i]: scores[i] for i in range(len(scores))}

    for key, value in sorted(
            node2score.items(), key=lambda d: d[1], reverse=True):
        print("{}: {}".format(key, value))
