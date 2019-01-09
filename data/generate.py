"""Generate a syntactic dataset."""

import numpy as np
import math

def generate_dataset(nodes, comparisons, noise):
    """Generates a syntactic dataset.

    The true ranking is that the higher the ID, the higher the
    ranking, i.e. the true ranking should be:
    1, 2, ..., N in ascending order.

    :param nodes: number of nodes in dataset
    :param comparisons: number of pairwise comparisons to generate
    :param noise: amount of noise in comparisons
    :returns:
    :rtype:
    """
    A = np.zeros([nodes, nodes], dtype=np.int)

    for i in range(comparisons):
        a = np.random.random_integers(0, nodes-1)
        b = np.random.random_integers(0, nodes-1)
        if a > b:
            A[a, b] += 1
        elif a < b:
            A[b, a] += 1

    return A
