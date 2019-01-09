"""Generates a syntactic data-set.

The true ranking is 1, 2, ..., N in ascending order.

Params:
  nodes: maximum number of nodes
  comparisons: total number of comparisons to make
  noise: level of noise in data (choose something < 0.5)

Usage:
  python data/generate_data.py --nodes 10 --comparisons 100 --noise 0.2
"""

import os
import numpy as np
import random

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate syntactic data.")

    parser.add_argument("--nodes", type=int, help="Maximum number of nodes", required=True)

    parser.add_argument(
        "--comparisons", type=int, help="Number of data-points to generate.", required=True)

    parser.add_argument(
        "--noise", type=float, default=0., help="Degree of noise in data.")

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, "syntactic")
    file_path = os.path.join(
        dir_path, "syntactic_{}_{}_{}.csv".format(args.nodes, args.comparisons, args.noise))

    with open(file_path, "w+") as fp:
        for i in range(args.comparisons):
            node_1 = np.random.random_integers(args.nodes)
            node_2 = np.random.choice(np.setdiff1d(range(1, args.nodes + 1), node_1))
            result = int(node_1 < node_2)
            if random.uniform(0, 1) < args.noise:
                result = 1 - result

            fp.write("{},{},{}\n".format(
                node_1,
                node_2,
                result
            ))
