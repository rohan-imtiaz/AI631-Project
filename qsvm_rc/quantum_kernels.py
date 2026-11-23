"""Fidelity kernels for the three feature input (Pu, ex, ey) on three qubits.

Two routes to the same kernel K(x, x') = |<0|S^dagger(x) S(x')|0>|^2:
1. exact_states / exact_gram: noiseless state vectors written out in NumPy. Three qubits means an
   8 dimensional state, so this is exact and fast. 00_setup checks it against Qiskit Statevector.
2. sampled_gram: the compute-uncompute circuit (unitary_overlap) sampled with shots on an Aer
   backend (ideal, noisy, or a fake device), exactly as on hardware.

Feature map specs are plain dicts, for example {"kind": "zz", "reps": 1, "entanglement": "linear"}.
kind is one of "angle" (one Ry per qubit), "z" (H then P(2x), no entanglers), "zz" (Havlicek ZZ map)
or "pauli_zzz" (Z, ZZ and ZZZ terms).
"""

from itertools import combinations

import numpy as np

NUMBER_OF_QUBITS = 3
ONE_QUBIT_NOISY_GATES = ["sx", "x"]
TWO_QUBIT_NOISY_GATES = ["cx"]
SIMULATION_BASIS = ["rz", "sx", "x", "cx"]


def spec_name(spec):
    if spec["kind"] == "angle":
        return "angle"
    kind = spec["kind"]
    reps = spec.get("reps", 1)
    entanglement = spec.get("entanglement", "linear")
    return f"{kind}_r{reps}_{entanglement}"


def entangling_groups(num_qubits, entanglement, block_size):
    """Qubit groups that get a multi-qubit phase term (order does not matter, the phases commute)."""
    qubits = list(range(num_qubits))
    if num_qubits < block_size:
        return []
    if entanglement == "full":
        return list(combinations(qubits, block_size))
    linear = [tuple(qubits[start:start + block_size]) for start in range(num_qubits - block_size + 1)]
    if entanglement == "circular" and num_qubits > block_size:
        wrap = tuple(qubits[num_qubits - block_size + 1:] + qubits[:1])
        return linear + [wrap]
    return linear


def _basis_bits(num_qubits):
    """bits[k, q] is the value of qubit q in basis state k (Qiskit little-endian order)."""
    return (np.arange(2 ** num_qubits)[:, None] >> np.arange(num_qubits)[None, :]) & 1


def _diagonal_phases(angles, spec, bits):
    """Phase of every basis state after one layer of the P / CX-P-CX blocks."""
    num_qubits = angles.shape[1]
    phases = 2 * angles @ bits.T
    entanglement = spec.get("entanglement", "linear")
    if spec["kind"] in ("zz", "pauli_zzz"):
        for i, j in entangling_groups(num_qubits, entanglement, 2):
            weight = 2 * (np.pi - angles[:, i]) * (np.pi - angles[:, j])
            parity = np.bitwise_xor(bits[:, i], bits[:, j])
            phases = phases + weight[:, None] * parity[None, :]
    if spec["kind"] == "pauli_zzz":
        for i, j, k in entangling_groups(num_qubits, entanglement, 3):
            weight = 2 * (np.pi - angles[:, i]) * (np.pi - angles[:, j]) * (np.pi - angles[:, k])
            parity = np.bitwise_xor(np.bitwise_xor(bits[:, i], bits[:, j]), bits[:, k])
            phases = phases + weight[:, None] * parity[None, :]
    return phases


def exact_states(angles, spec):
    """Noiseless encoded states, shape (number of points, 2 ** qubits)."""
    angles = np.atleast_2d(np.asarray(angles, dtype = float))
    count, num_qubits = angles.shape
    bits = _basis_bits(num_qubits)

    if spec["kind"] == "angle":
        cosines = np.cos(angles / 2)[:, None, :]
        sines = np.sin(angles / 2)[:, None, :]
        amplitudes = np.where(bits[None, :, :] == 1, sines, cosines).prod(axis = 2)
        return amplitudes.astype(complex)

    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    hadamard_all = np.array([[1]])
    for _ in range(num_qubits):
        hadamard_all = np.kron(hadamard_all, hadamard)

    phase_factors = np.exp(1j * _diagonal_phases(angles, spec, bits))
    states = np.zeros((count, 2 ** num_qubits), dtype = complex)
    states[:, 0] = 1
    for _ in range(spec.get("reps", 1)):
        states = states @ hadamard_all.T
        states = states * phase_factors
    return states


def exact_gram(spec, angles_a, angles_b = None):
    states_a = exact_states(angles_a, spec)
    states_b = states_a if angles_b is None else exact_states(angles_b, spec)
    return np.abs(np.conj(states_a) @ states_b.T) ** 2


def angle_closed_form_gram(angles_a, angles_b = None):
    """prod_j cos^2((x_j - x'_j) / 2), the synopsis equation (5)."""
    angles_b = angles_a if angles_b is None else angles_b
    return np.prod(np.cos((angles_a[:, None, :] - angles_b[None, :, :]) / 2) ** 2, axis = 2)


def global_depolarizing_fit(noisy_gram, exact_gram_values, num_qubits = NUMBER_OF_QUBITS):
    """Least squares lambda in K_noisy = lambda K_exact + (1 - lambda) / 2^n, over the off-diagonal entries."""
    mask = np.logical_not(np.eye(noisy_gram.shape[0], dtype = bool)) if noisy_gram.shape[0] == noisy_gram.shape[1] else np.ones(noisy_gram.shape, dtype = bool)
    floor = 1 / 2 ** num_qubits
    centred_exact = exact_gram_values[mask] - floor
    centred_noisy = noisy_gram[mask] - floor
    return float(np.sum(centred_exact * centred_noisy) / np.sum(centred_exact ** 2))


# Qiskit circuits (imported lazily so the NumPy part runs without Qiskit)

def build_feature_map(spec, num_qubits = NUMBER_OF_QUBITS):
    """Qiskit circuit S(x) with a ParameterVector named x."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import pauli_feature_map, zz_feature_map

    kind = spec["kind"]
    reps = spec.get("reps", 1)
    entanglement = spec.get("entanglement", "linear")
    if kind == "angle":
        parameters = ParameterVector("x", num_qubits)
        circuit = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            circuit.ry(parameters[qubit], qubit)
        return circuit
    if kind == "z":
        return pauli_feature_map(num_qubits, reps = reps, paulis = ["Z"])
    if kind == "zz":
        return zz_feature_map(num_qubits, reps = reps, entanglement = entanglement)
    if kind == "pauli_zzz":
        return pauli_feature_map(num_qubits, reps = reps, entanglement = entanglement, paulis = ["Z", "ZZ", "ZZZ"])
    raise ValueError(f"Unknown feature map kind {kind}")


def overlap_template(feature_map):
    """Measured, still parameterised S^dagger(p2) S(p1) circuit (NB08 unitary_overlap)."""
    from qiskit.circuit.library import unitary_overlap

    overlap = unitary_overlap(feature_map, feature_map)
    overlap.measure_all()
    return overlap


def overlap_parameters(circuit):
    """p1 and p2 parameters sorted by index (bind by name, never by position)."""
    first = sorted([parameter for parameter in circuit.parameters if parameter.name.startswith("p1")], key = lambda parameter: parameter.index)
    second = sorted([parameter for parameter in circuit.parameters if parameter.name.startswith("p2")], key = lambda parameter: parameter.index)
    return first, second


def pair_list(angles_a, angles_b = None, include_diagonal = False):
    """Index pairs to evaluate. Symmetric case: upper triangle only (diagonal optional)."""
    if angles_b is None:
        offset = 0 if include_diagonal else 1
        return [(i, j) for i in range(len(angles_a)) for j in range(i + offset, len(angles_a))]
    return [(i, j) for i in range(len(angles_a)) for j in range(len(angles_b))]


def sampled_gram(transpiled_overlap, backend, angles_a, angles_b = None, shots = 2048, seed = 0,
                 include_diagonal = False, chunk_size = 50000):
    """Kernel estimated as the fraction of all-zero outcomes, one circuit per pair.

    transpiled_overlap must come from overlap_template and be transpiled for `backend`.
    Symmetric case: the diagonal is pinned to 1 unless include_diagonal is True.
    """
    angles_a = np.asarray(angles_a, dtype = float)
    symmetric = angles_b is None
    angles_b_used = angles_a if symmetric else np.asarray(angles_b, dtype = float)
    pairs = pair_list(angles_a, None if symmetric else angles_b_used, include_diagonal)
    first, second = overlap_parameters(transpiled_overlap)
    zeros = "0" * len(first)

    values = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start:start + chunk_size]
        binds = {first[k]: [float(angles_a[i, k]) for i, _ in chunk] for k in range(len(first))}
        binds.update({second[k]: [float(angles_b_used[j, k]) for _, j in chunk] for k in range(len(second))})
        result = backend.run(transpiled_overlap, shots = shots, parameter_binds = [binds],
                             seed_simulator = seed + start).result()
        values.extend(result.get_counts(position).get(zeros, 0) / shots for position in range(len(chunk)))

    if symmetric:
        gram = np.eye(len(angles_a))
        for (i, j), value in zip(pairs, values):
            gram[i, j] = value
            gram[j, i] = value
        return gram
    gram = np.zeros((len(angles_a), len(angles_b_used)))
    for (i, j), value in zip(pairs, values):
        gram[i, j] = value
    return gram


def noise_model(one_qubit_error = 0, two_qubit_error = 0, readout_error = 0):
    """Depolarizing gate noise on sx, x (one qubit) and cx (two qubit), plus symmetric readout error."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    model = NoiseModel()
    if one_qubit_error > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(one_qubit_error, 1), ONE_QUBIT_NOISY_GATES)
    if two_qubit_error > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), TWO_QUBIT_NOISY_GATES)
    if readout_error > 0:
        model.add_all_qubit_readout_error(ReadoutError([[1 - readout_error, readout_error],
                                                        [readout_error, 1 - readout_error]]))
    return model


def noisy_simulator(one_qubit_error = 0, two_qubit_error = 0, readout_error = 0):
    """Density matrix AerSimulator (exact noisy probabilities, then shots are sampled)."""
    from qiskit_aer import AerSimulator

    model = noise_model(one_qubit_error, two_qubit_error, readout_error)
    if model.is_ideal():
        return AerSimulator(max_parallel_experiments = 0, max_parallel_threads = 0)
    # max_parallel_experiments = 0 spreads the bound circuits over every CPU core
    return AerSimulator(noise_model = model, method = "density_matrix",
                        max_parallel_experiments = 0, max_parallel_threads = 0)


def transpile_for_simulation(circuit):
    """Rewrite into rz, sx, x, cx so the gate noise attaches to real gate names."""
    from qiskit import transpile

    return transpile(circuit, basis_gates = SIMULATION_BASIS, optimization_level = 1)


def fake_device_backend(preferred = ("FakeFez", "FakeMarrakesh", "FakeTorino", "FakeSherbrooke")):
    """A fake IBM device (snapshot noise model) from qiskit_ibm_runtime.fake_provider."""
    from qiskit_ibm_runtime import fake_provider

    for name in preferred:
        if hasattr(fake_provider, name):
            return getattr(fake_provider, name)()
    raise ImportError("No fake backend found in qiskit_ibm_runtime.fake_provider")


def circuit_resources(circuit):
    """Qubits actually used, depth, two-qubit gate count and total gate count of a circuit."""
    active_qubits = set()
    two_qubit_gates = 0
    total_gates = 0
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in ("barrier", "measure"):
            continue
        total_gates = total_gates + 1
        for qubit in instruction.qubits:
            active_qubits.add(circuit.find_bit(qubit).index)
        if instruction.operation.num_qubits == 2:
            two_qubit_gates = two_qubit_gates + 1
    return {
        "qubits_used": len(active_qubits),
        "depth": circuit.depth(lambda instruction: instruction.operation.name != "barrier"),
        "two_qubit_gates": two_qubit_gates,
        "total_gates": total_gates,
    }
