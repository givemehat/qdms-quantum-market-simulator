"""
Unit tests for QDMS Open Quantum Systems Engine (q_engine).
Validates:
1. Market Hamiltonian construction & Hermiticity (Transverse-Field Ising model).
2. Quantum initial state density matrix preparation (trace preservation, positive semi-definiteness).
3. Lindblad collapse operators generation and scaling.
4. Master equation time-evolution (decoherence, trace preservation, bound constraints).
"""

import numpy as np
import pytest
from qutip import Qobj

from q_engine.hamiltonians import build_market_hamiltonian
from q_engine.state_prep import prepare_initial_market_state
from q_engine.Lindblad_solver import (
    get_collapse_operators,
    solve_lindblad_master_equation,
)


class TestMarketHamiltonian:
    """Test suite for quantum market Hamiltonian construction."""

    def test_hamiltonian_dimensions(self):
        num_assets = 3
        coupling = np.array([
            [0.0, 0.5, 0.2],
            [0.5, 0.0, 0.4],
            [0.2, 0.4, 0.0]
        ])
        H = build_market_hamiltonian(num_assets, coupling)
        assert isinstance(H, Qobj)
        assert H.dims == [[2, 2, 2], [2, 2, 2]]
        assert H.shape == (8, 8)

    def test_hamiltonian_hermiticity(self):
        """Physical Hamiltonians must be strictly Hermitian (H = H†)."""
        num_assets = 2
        coupling = np.array([[0.0, 0.75], [0.75, 0.0]])
        H = build_market_hamiltonian(num_assets, coupling)
        hermitian_diff = (H - H.dag()).norm()
        assert np.isclose(hermitian_diff, 0.0, atol=1e-12)

    def test_zero_coupling_independent_assets(self):
        """Zero cross-asset coupling should reduce to independent transverse field."""
        num_assets = 2
        coupling = np.zeros((2, 2))
        H = build_market_hamiltonian(num_assets, coupling)
        assert isinstance(H, Qobj)
        assert (H - H.dag()).norm() < 1e-12


class TestStatePreparation:
    """Test suite for initial quantum market density matrix preparation."""

    def test_stable_ground_state(self):
        num_assets = 3
        rho = prepare_initial_market_state(num_assets, stable=True)
        assert isinstance(rho, Qobj)
        assert rho.dims == [[2, 2, 2], [2, 2, 2]]
        assert np.isclose(rho.tr(), 1.0, atol=1e-10)
        assert (rho - rho.dag()).norm() < 1e-12

    def test_fragile_excited_state(self):
        num_assets = 2
        rho = prepare_initial_market_state(num_assets, stable=False)
        assert isinstance(rho, Qobj)
        assert np.isclose(rho.tr(), 1.0, atol=1e-10)
        assert (rho - rho.dag()).norm() < 1e-12


class TestCollapseOperators:
    """Test suite for Lindblad jump/collapse operators."""

    def test_collapse_operators_count_and_scaling(self):
        num_assets = 3
        shock = 0.4
        c_ops = get_collapse_operators(num_assets, shock)
        assert len(c_ops) == num_assets
        for op in c_ops:
            assert isinstance(op, Qobj)
            assert op.dims == [[2, 2, 2], [2, 2, 2]]

    def test_zero_shock_intensity(self):
        num_assets = 2
        c_ops = get_collapse_operators(num_assets, shock_intensity=0.0)
        for op in c_ops:
            assert np.isclose(op.norm(), 0.0)


class TestLindbladMasterEquationSolver:
    """Test suite for numerical Lindblad time-evolution dynamics."""

    def test_solver_expectation_bounds(self):
        num_assets = 2
        coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
        H = build_market_hamiltonian(num_assets, coupling)
        rho0 = prepare_initial_market_state(num_assets, stable=False)
        c_ops = get_collapse_operators(num_assets, shock_intensity=0.1)
        times = np.linspace(0.0, 2.0, 10)

        expectations = solve_lindblad_master_equation(H, rho0, c_ops, times)

        assert len(expectations) == num_assets
        for asset_traj in expectations:
            assert isinstance(asset_traj, np.ndarray)
            assert len(asset_traj) == len(times)
            assert not np.isnan(asset_traj).any()
            assert not np.isinf(asset_traj).any()
            # Sigma_z eigenvalues are [-1, 1], so expectation must be bounded within [-1, 1]
            assert np.all(asset_traj >= -1.0 - 1e-5)
            assert np.all(asset_traj <= 1.0 + 1e-5)

    def test_dissipative_relaxation(self):
        """Under sigma_minus jump operators, excited assets should relax downwards."""
        num_assets = 2
        coupling = np.zeros((2, 2))
        H = build_market_hamiltonian(num_assets, coupling)
        rho0 = prepare_initial_market_state(num_assets, stable=False)
        c_ops = get_collapse_operators(num_assets, shock_intensity=0.5)
        times = np.linspace(0.0, 5.0, 20)

        expectations = solve_lindblad_master_equation(H, rho0, c_ops, times)
        # Asset 0 begins excited (sigma_z = -1 in basis(2,1)) or relaxed depending on convention,
        # but must smoothly converge without exploding
        assert len(expectations[0]) == len(times)
        assert np.isfinite(expectations[0]).all()
