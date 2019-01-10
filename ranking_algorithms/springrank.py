"""SpringRank

SpringRank is a physically inspired model and an efficient algorithm
to infer hierarchical rankings of nodes in directed networks.

http://advances.sciencemag.org/content/4/7/eaar8260

Usage:
  python ranking_algorithms/springrank.py --path data/blueprint-rookie-data/
"""

from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np

import csv
import argparse

import os
import pickle


def get_adj_matrix(file_path, num_nodes):
    """Parses a data-set."""

    N = len(idx2node)

    A = np.zeros([N, N])

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for winner, loser in csv_reader:
            winner = int(winner)
            loser = int(loser)
            A[winner, loser] += 1

    return A


def spring_rank(adj_matrix, alpha=0., reg_rl=1.0, int_rl=1.0):
    """Computes the SpringRank model.

    SpringRank is computed via solving a set of linear equations,
    which makes it fast. SpringRank works by imagining springs between
    the various nodes, and the ranking solution is the configuration
    where the total energy is lowest.

    :param adj_matrix: weighted network adjacency matrix
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

    return rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", help="file path of data-set.", required=True)

    args = parser.parse_args()

    idx2node_path = os.path.join(args.file, "idx2node.pkl")

    with open(idx2node_path, "rb") as f:
        idx2node = pickle.load(f)

    data_path = os.path.join(args.file, "data.csv")

    num_nodes = len(idx2node)

    adj_matrix = get_adj_matrix(data_path, num_nodes)

    print("Total Nodes: ", num_nodes)

    rankings = spring_rank(adj_matrix)

    node2score = {idx2node[i]: rankings[i] for i in range(len(rankings))}

    for key, value in sorted(
            node2score.items(), key=lambda d: d[1], reverse=True):
        print("{}: {}".format(key, value))
