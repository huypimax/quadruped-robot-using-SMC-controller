#!/usr/bin/env python3
"""Quick sanity checks for quadruped_dynamics.

This isn't a full unit test framework; it's a tiny script you can run to verify:
- shapes
- M is symmetric
- forward_dynamics runs
"""

import numpy as np

from quadruped_dynamics import quadruped_dynamics, forward_dynamics, tau_limit_vector, clip_tau


def main() -> None:
    q = np.zeros(12)
    # A pose similar to your Matlab q0, repeated 4 legs.
    q[1::3] = 0.8
    q[2::3] = -1.5

    dq = np.zeros(12)
    tau = np.zeros(12)

    M, C, G = quadruped_dynamics(q, dq)
    assert M.shape == (12, 12)
    assert C.shape == (12, 12)
    assert G.shape == (12,)

    # Symmetry check (numerical)
    sym_err = np.max(np.abs(M - M.T))
    print(f"max|M-M^T| = {sym_err:.3e}")

    ddq = forward_dynamics(q, dq, tau)
    assert ddq.shape == (12,)
    print(f"ddq (first leg) = {ddq[0:3]}")

    tau_max = tau_limit_vector()
    tau2 = clip_tau(np.ones(12) * 1e6, tau_max)
    print(f"tau clipped = {tau2[0:3]} (expected [20,55,55])")

    print("OK")


if __name__ == "__main__":
    main()
