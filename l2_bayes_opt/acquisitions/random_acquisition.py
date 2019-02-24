# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import numpy as np

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition


class Random(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable],
                 jitter: np.float64 = np.float64(0))-> None:
        pass

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        """
        return np.random.random((x.shape[0], 1))

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.
        :param x: locations where the evaluation with gradients is done.
        """
        raise NotImplementedError()

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False
