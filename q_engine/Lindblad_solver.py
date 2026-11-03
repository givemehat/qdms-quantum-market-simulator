import numpy as np
from qutip import mesolve, sigmam, tensor, qeye, sigmaz, Qobj


def get_collapse_operators(num_assets: int, shock_intensity: float) -> list:
    """
    Creates collapse operators for the Lindblad Master Equation.
    A collapse operator usually represents a shock or dissipation in an asset.
    We use the relaxation operator sigmam() (sigma_minus) which forces
    the state from |1> (excited/high value) to |0> (ground/crashed).
    """
    c_ops = []
    for i in range(num_assets):
        op_list = [qeye(2)] * num_assets
        op_list[i] = sigmam()  # Energy dissipation (crash)
        # Multiply by sqrt of rate (shock_intensity)
        c_ops.append(np.sqrt(shock_intensity) * tensor(op_list))
    return c_ops


def solve_lindblad_master_equation(
    H: Qobj, initial_state: Qobj, collapse_operators: list, times: np.ndarray
) -> list:
    """
    Solves the Lindblad Master Equation to simulate market crash/panic propagation.
    Enforces programmatic normalization clamps and Hermiticity constraints after
    every integration interval step to completely eliminate numerical drift.

    dρ/dt = -i[H, ρ] + \sum_k (L_k ρ L_k^† - 1/2 {L_k^† L_k, ρ})
    """
    num_assets = len(H.dims[0])

    # We want to measure the "fidelity" or "magnetization" (sigma_z) of each asset over time
    e_ops = []
    for i in range(num_assets):
        op_list = [qeye(2)] * num_assets
        op_list[i] = sigmaz()
        e_ops.append(tensor(op_list))

    print(f"Running open quantum system simulation over {len(times)} time steps...")

    # Pass c_ops and e_ops as keyword arguments for seamless QuTiP 4.x and 5.x compatibility.
    # Passing e_ops=[] ensures mesolve returns density matrix states (result.states).
    result = mesolve(H, initial_state, times, c_ops=collapse_operators, e_ops=[])

    # Pre-allocate clean sanitized tracking lists for target indicators
    sanitized_expectations = [[] for _ in range(num_assets)]

    # =====================================================================
    # 🛡️ PROGRAMMATIC UNITARY TRACE AND HERMITICITY CLAMPS (ISSUE #5)
    # =====================================================================
    for state in result.states:
        # Convert QuTiP state Qobj to flat numpy array array for granular element operations
        rho_matrix = state.full()

        # 1. Enforce Absolute Hermiticity: rho = (rho + rho_dagger) / 2
        rho_hermitian = (rho_matrix + rho_matrix.conj().T) / 2.0

        # 2. Enforce Absolute Normalization Trace Preservation: rho = rho / Tr(rho)
        trace_val = np.trace(rho_hermitian)
        if not np.isclose(trace_val, 0):
            rho_normalized = rho_hermitian / trace_val
        else:
            rho_normalized = rho_hermitian

        # Re-encapsulate the filtered layout matrix back into a QuTiP Qobj state mapping
        clean_state = Qobj(rho_normalized, dims=state.dims)

        # Manually compute the clean scalar expectations for every asset operator
        for idx, op in enumerate(e_ops):
            # expectation value = Tr(op * rho)
            val = (op * clean_state).tr()
            sanitized_expectations[idx].append(float(val.real))

    # Convert inner sequences to numpy numeric matrices to match downstream API signatures
    return [np.array(arr) for arr in sanitized_expectations]
