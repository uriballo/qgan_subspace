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
"""Cost and Fidelity Functions using PyTorch."""

import torch
from config import CFG

# Set seed for reproducibility
torch.manual_seed(42)

def braket(*args) -> torch.Tensor:
    """
    Calculate the braket (inner product) <bra|op1|op2|...|ket> between quantum states.
    All inputs are expected to be PyTorch tensors.
    """
    bra, *ops, ket = args

    for op in ops:
        ket = torch.matmul(op, ket)
    
    # Use .mH for the conjugate transpose (Hermitian) of a matrix
    return torch.matmul(bra.mH, ket)


def compute_cost(dis, final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Wasserstein-like cost function.
    This function must return a scalar tensor for backpropagation.
    """
    # Assumes dis.get_dis_matrices_rep() will be updated to return tensors
    A, B, psi, phi = dis.get_dis_matrices_rep()

    # braket function now returns a tensor, so we squeeze it to a scalar
    term1 = braket(final_gen_state, A, final_gen_state).squeeze()
    term2 = braket(final_target_state, B, final_target_state).squeeze()
    term3 = braket(final_gen_state, B, final_target_state).squeeze()
    term4 = braket(final_target_state, A, final_gen_state).squeeze()
    term5 = braket(final_gen_state, A, final_target_state).squeeze()
    term6 = braket(final_target_state, B, final_gen_state).squeeze()
    term7 = braket(final_gen_state, B, final_gen_state).squeeze()
    term8 = braket(final_target_state, A, final_target_state).squeeze()

    # The density matrices rho = |psi⟩⟨psi|
    rho_target = torch.matmul(final_target_state, final_target_state.mH)
    rho_gen = torch.matmul(final_gen_state, final_gen_state.mH)

    # Trace terms: Tr(rho * op)
    psiterm = torch.trace(torch.matmul(rho_target, psi))
    phiterm = torch.trace(torch.matmul(rho_gen, phi))

    # Regularization term
    reg_inner = CFG.cst1 * term1 * term2 - CFG.cst2 * (term3 * term4 + term5 * term6) + CFG.cst3 * term7 * term8
    regterm = (CFG.lamb / torch.e) * reg_inner

    # The final loss must be a real-valued scalar tensor
    loss = (psiterm - phiterm - regterm).real
    
    return loss


def compute_fidelity(final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> float:
    """
    Calculate the fidelity |<target|gen>|^2 between two states.
    Returns a float, as this is typically used for monitoring, not for gradients.
    """
    braket_result = braket(final_target_state, final_gen_state)
    # .item() extracts the scalar value from the tensor
    return torch.abs(braket_result).pow(2).item()


def compute_fidelity_and_cost(dis, final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Calculate the fidelity (float) and cost function (tensor)."""
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    cost = compute_cost(dis, final_target_state, final_gen_state)
    return fidelity, cost