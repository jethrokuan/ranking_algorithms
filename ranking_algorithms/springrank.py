"""SpringRank

SpringRank is a physically inspired model and an efficient algorithm
to infer hierarchical rankings of nodes in directed networks.

http://advances.sciencemag.org/content/4/7/eaar8260

Usage:
  python ranking_algorithms/springrank.py --file data/blueprint-rookie-data/data.csv
"""

from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np

import csv
import argparse


def parse_file(file_path):
    """Parses a data-set."""

    nodes = set([])

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for winner, loser in csv_reader:
            nodes.add(winner)
            nodes.add(loser)

    idx2node = list(nodes)
    node2idx = {idx2node[i]: i for i in range(len(idx2node))}
    N = len(idx2node)

    A = np.zeros([N, N])

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for winner, loser in csv_reader:
            winner_idx = node2idx[winner]
            loser_idx = node2idx[loser]
            A[winner_idx, loser_idx] += 1

    return idx2node, A


def spring_rank(adj_matrix, idx2node, alpha=0., reg_rl=1.0, int_rl=1.0):
    """Computes the SpringRank model.

    SpringRank is computed via solving a set of linear equations,
    which makes it fast. SpringRank works by imagining springs between
    the various nodes, and the ranking solution is the configuration
    where the total energy is lowest.

    :param adj_matrix: weighted network adjacency matrix
    :param idx2node: mapping from index to node
    :param alpha: controls the impact of the regularization term
    :param reg_rl: regularization spring's rest length
    :param int_rl: interaction spring's rest length
    :returns:
    :rtype: N-dimensional array, indices representing the node's indices
      in ordering adj_matrix

    """
    N = adj_matrix.shape[0]
    k_in = np.sum(adj_matrix, 0)
    k_out = np.sum(adj_matrix, 1)

    D1 = np.zeros_like(adj_matrix)
    D2 = np.zeros_like(adj_matrix)

    for i in range(N):
        D1[i, i] = k_out[i] + k_in[i]
        D2[i, i] = int_rl * (k_out[i] - k_in[i])

    B = np.ones(N) * alpha * reg_rl + np.dot(D2, np.ones(N))
    A = alpha * np.eye(N) + D1 - (adj_matrix + adj_matrix.T)
    A = csr_matrix(A)
    rank = linalg.bicgstab(A, B)[0]
    node2score = {idx2node[i]: rank[i] for i in range(len(rank))}

    return node2score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", help="file path of data-set.", required=True)

    args = parser.parse_args()

    idx2node, adj_matrix = parse_file(args.file)

    print("Total Nodes: ", len(idx2node))

    res = spring_rank(adj_matrix, idx2node)

    for key, value in sorted(res.items(), key=lambda d: d[1], reverse=True):
        print("{}: {}".format(key, value))
