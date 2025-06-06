# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimizer file"""

from config import CFG
from tools.data.data_reshapers import _flatten, _unflatten


class MomentumOptimizer:
    """
    gradient descent with momentum, given an objective function to be minimized.
    the update formula from
    On the importance of initialization and momentum in deep learning
    http://proceedings.mlr.press/v28/sutskever13.pdf

    v_{t+1} = miu * v_{t} - eta * grad(theta_t)
    theta_{t+1} = theta_{t} + v_{t+1}

    eta:= learning rate > 0
    miu:= momentum coefficient [0,1]
    """

    def __init__(self, eta=CFG.l_rate, miu=CFG.momentum_coeff):
        self.miu = miu
        self.eta = eta
        self.v = None

    def compute_grad(self, theta, grad_list, min_or_max):
        grad_list = _flatten(grad_list)
        theta_tmp = _flatten(theta)

        if min_or_max == "min":
            if self.v is None:
                self.v = [-self.eta * g for g in grad_list]
            else:
                self.v = [self.miu * v - self.eta * g for v, g in zip(self.v, grad_list)]

            new_theta = [theta_i + v for theta_i, v in zip(theta_tmp, self.v)]
        else:
            if self.v is None:
                self.v = [self.eta * g for g in grad_list]
            else:
                self.v = [self.miu * v + self.eta * g for v, g in zip(self.v, grad_list)]

            new_theta = [theta_i + v for theta_i, v in zip(theta_tmp, self.v)]

        new_theta = _unflatten(new_theta, theta)

        return new_theta
