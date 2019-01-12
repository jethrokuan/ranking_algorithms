"""SpringRank

SpringRank is a physically inspired model and an efficient algorithm
to infer hierarchical rankings of nodes in directed networks.

http://advances.sciencemag.org/content/4/7/eaar8260

Usage:
  python ranking_algorithms/springrank.py --path data/blueprint-rookie-data/
"""

import argparse
import csv
import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix, linalg


def get_adj_matrix(file_path, node_count):
    adj = np.zeros([node_count, node_count])

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for winner, loser in csv_reader:
            winner = int(winner)
            loser = int(loser)
            adj_matrix[winner, loser] += 1

    return adj


def spring_rank(adj, alpha=0., reg_rl=1.0, int_rl=1.0):
    """Computes the SpringRank model.

    SpringRank is computed via solving a set of linear equations,
    which makes it fast. SpringRank works by imagining springs between
    the various nodes, and the ranking solution is the configuration
    where the total energy is lowest.

    Args:
      adj: weighted network adjacency matrix
      alpha: controls the impact of the regularization term (Default value = 0.)
      reg_rl: regularization spring's rest length (Default value = 1.0)
      int_rl: interaction spring's rest length (Default value = 1.0)

    Returns:
      n-d arr: array with indices representing the node's indices

    """
    n = adj.shape[0]
    k_in = np.sum(adj, 0)
    k_out = np.sum(adj, 1)

    d1 = np.zeros_like(adj)
    d2 = np.zeros_like(adj)

    for i in range(n):
        d1[i, i] = k_out[i] + k_in[i]
        d2[i, i] = int_rl * (k_out[i] - k_in[i])

    b = np.ones(n) * alpha * reg_rl + np.dot(d2, np.ones(n))
    a = alpha * np.eye(N) + d1 - (adj + adj.T)
    a = csr_matrix(a)
    rank = linalg.bicgstab(a, b)[0]

    return rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", help="file path of data-set.", required=True)

    args = parser.parse_args()

    idx2node_path = os.path.join(args.file, "idx2node.pkl")

    with open(idx2node_path, "rb") as fp:
        idx2node = pickle.load(fp)

    data_path = os.path.join(args.file, "data.csv")

    num_nodes = len(idx2node)

    adj_matrix = get_adj_matrix(data_path, num_nodes)

    print("Total Nodes: ", num_nodes)

    rankings = spring_rank(adj_matrix)

    node2score = {idx2node[i]: rankings[i] for i in range(len(rankings))}

    for key, value in sorted(
            node2score.items(), key=lambda d: d[1], reverse=True):
        print("{}: {}".format(key, value))
