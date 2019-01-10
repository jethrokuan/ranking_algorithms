"""Data Utilities."""

import csv

import os
import errno


def mkdirs(newdir, mode="0777"):
    try:
        os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise
