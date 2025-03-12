#!/usr/bin/env python

"""
utils.py: some public tool functions

"""

import os
import pickle
from collections.abc import Iterable

import numpy as np


def _flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, other): each element of the Iterable may itself be an iterable object

    Yields:
        other: elements of x in depth-first order
    """
    it = x
    for x in it:
        # if (isinstance(x, collections.Iterable) and
        if isinstance(x, Iterable) and not isinstance(x, str):
            for y in _flatten(x):
                yield y
        else:
            yield x


def _unflatten(flat, prototype):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(prototype, np.ndarray):
        idx = prototype.size
        res = np.array(flat)[:idx].reshape(prototype.shape)
        return res

    # if isinstance(prototype, collections.Iterable):
    if isinstance(prototype, Iterable):
        res = []
        for x in prototype:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat

    raise TypeError("Unsupported type in the model: {}".format(type(prototype)))


def unflatten(flat, prototype):
    """Wrapper for :func:`_unflatten`."""
    # pylint:disable=len-as-condition
    result, tail = _unflatten(flat, prototype)
    if len(tail) != 0:
        raise ValueError("Flattened iterable has more elements than the model.")
    return result


def train_log(param, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        file.write(param)


def load_model(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "rb") as qc:
        model = pickle.load(qc)
    return model


def save_model(gen, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb+") as file:
        pickle.dump(gen, file)


def save_fidelity_loss(fidelities_history, losses_history, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        np.save(f, fidelities_history)
        np.save(f, losses_history)


def save_theta(gen, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    array_angle = np.zeros(len(gen.qc.gates))
    for i in range(len(gen.qc.gates)):
        array_angle[i] = gen.qc.gates[i].angle
    np.savetxt(file_path, array_angle)
