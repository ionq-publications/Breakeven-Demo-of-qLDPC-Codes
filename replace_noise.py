import re
from collections import defaultdict

import stim


def replace_depol2_after_entangling(circuit, noise_dict):
    """Replace DEPOLARIZE2 noise following CX/CZ gates using per-pair noise rates.

    Parameters
    ----------
    circuit : stim.Circuit
        Input stim circuit.
    noise_dict : dict[(int, int), float]
        Maps qubit pairs (q0, q1) to a DEPOLARIZE2 noise parameter.
        Pairs not in the dict keep their original noise value.

    Returns
    -------
    stim.Circuit
        New circuit with DEPOLARIZE2 instructions replaced where applicable.
    """
    out = stim.Circuit()
    # Track qubit pairs from the most recent CX/CZ; cleared on the next CX/CZ
    entangled_pairs = set()
    i_start = 0

    for i_curr, op in enumerate(circuit):
        if op.name in ("CX", "CZ"):
            targets = [t.value for t in op.targets_copy()]
            entangled_pairs = {
                (targets[j], targets[j + 1]) for j in range(0, len(targets), 2)
            }

        elif op.name == "DEPOLARIZE2":
            targets = [t.value for t in op.targets_copy()]
            dep_pairs = [
                (targets[j], targets[j + 1]) for j in range(0, len(targets), 2)
            ]

            # Only replace if at least one pair comes from a recent CX/CZ
            if entangled_pairs and any(p in entangled_pairs for p in dep_pairs):
                if i_curr > i_start:
                    out += circuit[i_start:i_curr]
                i_start = i_curr + 1

                original_p = op.gate_args_copy()[0]
                pairs_by_noise = defaultdict(list)
                for pair in dep_pairs:
                    if pair in entangled_pairs:
                        p = noise_dict.get(pair, original_p)
                    else:
                        p = original_p
                    pairs_by_noise[p].append(pair)

                for p, pairs in pairs_by_noise.items():
                    flat = [q for pair in pairs for q in pair]
                    out.append("DEPOLARIZE2", flat, p)

        else:
            # Any other instruction invalidates entangled pairs it touches
            touched = {t.value for t in op.targets_copy()}
            entangled_pairs = {
                p
                for p in entangled_pairs
                if p[0] not in touched and p[1] not in touched
            }

    # Flush remaining instructions
    num_instructions = len(circuit)
    if i_start < num_instructions:
        out += circuit[i_start:num_instructions]

    return out


def build_qubit_drb_map(iondrb, data, ancillae):
    """Given lists of ions comprising data and ancillae,
    build a lookup table from iondrb, with tuples of qubits
    as keys instead of ions."""
    qbdrb = {}
    for i, d in enumerate(data):  # i,j are qubits
        for j, a in enumerate(ancillae):  # d, a are ions
            k = j + len(data)  # Ancillae shifted to start AFTER data qubit indices
            infidelity = iondrb[(d, a)]
            q0 = min(i, k)
            q1 = max(i, k)
            qbdrb[(q0, q1)] = infidelity
            qbdrb[(q1, q0)] = infidelity

    return qbdrb


def replace_depol2_after_sequence(circuit, gate_sequence, noise_dict):
    """Replace DEPOLARIZE2 noise on pairs that completed a specific gate sequence.

    Tracks qubit pairs through a sequence of gates. A pair is eligible for
    replacement only if every gate in ``gate_sequence`` was applied (in order)
    to both qubits of the pair, with no other instruction touching either qubit
    in between. The first gate in the sequence must be entangling (CX/CZ).

    Parameters
    ----------
    circuit : stim.Circuit
        Input stim circuit.
    gate_sequence : tuple[str, ...]
        Ordered gate names to match, e.g. ``('CZ', 'S', 'H')``.
    noise_dict : dict[(int, int), float]
        Maps qubit pairs (q0, q1) to a DEPOLARIZE2 noise parameter.
        Matched pairs not in the dict keep their original noise value.

    Returns
    -------
    stim.Circuit
        New circuit with DEPOLARIZE2 instructions replaced where applicable.
    """
    entangling_gate = gate_sequence[0]
    follow_gates = gate_sequence[1:]

    out = stim.Circuit()
    # Per-pair progress through follow_gates (index into follow_gates)
    pair_progress = {}
    i_start = 0

    for i_curr, op in enumerate(circuit):
        if op.name == entangling_gate:
            targets = [t.value for t in op.targets_copy()]
            for j in range(0, len(targets), 2):
                pair = (targets[j], targets[j + 1])
                pair_progress[pair] = 0

        elif op.name == "DEPOLARIZE2":
            targets = [t.value for t in op.targets_copy()]
            dep_pairs = [
                (targets[j], targets[j + 1]) for j in range(0, len(targets), 2)
            ]

            matched = {
                p for p in dep_pairs if pair_progress.get(p) == len(follow_gates)
            }

            if matched:
                if i_curr > i_start:
                    out += circuit[i_start:i_curr]
                i_start = i_curr + 1

                original_p = op.gate_args_copy()[0]
                pairs_by_noise = defaultdict(list)
                for pair in dep_pairs:
                    if pair in matched:
                        p = noise_dict.get(pair, original_p)
                    else:
                        p = original_p
                    pairs_by_noise[p].append(pair)

                for p, pairs in pairs_by_noise.items():
                    flat = [q for pair in pairs for q in pair]
                    out.append("DEPOLARIZE2", flat, p)

            for pair in dep_pairs:
                pair_progress.pop(pair, None)

        else:
            touched = {t.value for t in op.targets_copy()}
            to_remove = []
            for pair, idx in pair_progress.items():
                hits = (pair[0] in touched) + (pair[1] in touched)
                if hits == 0:
                    continue
                if (
                    hits == 2
                    and idx < len(follow_gates)
                    and op.name == follow_gates[idx]
                ):
                    pair_progress[pair] = idx + 1
                else:
                    to_remove.append(pair)
            for pair in to_remove:
                del pair_progress[pair]

    num_instructions = len(circuit)
    if i_start < num_instructions:
        out += circuit[i_start:num_instructions]

    return out


def replace_all_noise(circuit, noise_2q, pz_idle, px_idle, pz_mcm, px_mcm, p1=None, pm=None):
    """Replace all noise operations in a circuit with specified parameters.

    Parameters
    ----------
    circuit : stim.Circuit
        Input stim circuit.
    noise_2q : dict[(int, int), float]
        Maps qubit pairs (q0, q1) to a DEPOLARIZE2 noise parameter.
    pz_idle : float
        Z_ERROR parameter for idle errors.
    px_idle : float
        X_ERROR parameter for idle errors.
    pz_mcm : float
        Z_ERROR parameter for mid-circuit measurement errors (immediately after MR).
    px_mcm : float
        X_ERROR parameter for mid-circuit measurement errors (after MR, Z_ERROR).
    p1 : float, optional
        DEPOLARIZE1 noise parameter. If None, DEPOLARIZE1 instructions are left unchanged.
    pm : float, optional
        Measurement error parameter. If given, replaces MR(p) with MR(pm).
    """
    _mr_re = re.compile(r"^MR\([\d.e+-]+\)")
    _depol1_re = re.compile(r"^DEPOLARIZE1\([\d.e+-]+\)")
    _depol2_re = re.compile(r"^DEPOLARIZE2\(([\d.e+-]+)\)\s+(.+)$")
    _zerr_re = re.compile(r"^Z_ERROR\([\d.e+-]+\)")
    _xerr_re = re.compile(r"^X_ERROR\([\d.e+-]+\)")

    lines = str(circuit).split("\n")
    out_lines = []
    prev_was_mr = False
    prev_was_mr_then_zerr = False

    for line in lines:
        m_d2 = _depol2_re.match(line)
        if m_d2:
            original_p = m_d2.group(1)
            targets_str = m_d2.group(2)
            parts = targets_str.split()
            # Check if any pair needs replacement
            needs_split = False
            for j in range(0, len(parts), 2):
                if (int(parts[j]), int(parts[j + 1])) in noise_2q:
                    needs_split = True
                    break
            if not needs_split:
                out_lines.append(line)
            else:
                pairs_by_noise = defaultdict(list)
                for j in range(0, len(parts), 2):
                    pair = (int(parts[j]), int(parts[j + 1]))
                    p = noise_2q.get(pair, original_p)
                    pairs_by_noise[p].append(f"{parts[j]} {parts[j + 1]}")
                for p, tgt_strs in pairs_by_noise.items():
                    out_lines.append(f"DEPOLARIZE2({p}) {' '.join(tgt_strs)}")
            prev_was_mr = False
            prev_was_mr_then_zerr = False

        elif p1 is not None and _depol1_re.match(line):
            targets = line.split(") ", 1)[1]
            out_lines.append(f"DEPOLARIZE1({p1}) {targets}")
            prev_was_mr = False
            prev_was_mr_then_zerr = False

        elif _zerr_re.match(line):
            targets = line.split(") ", 1)[1]
            p = pz_mcm if prev_was_mr else pz_idle
            out_lines.append(f"Z_ERROR({p}) {targets}")
            prev_was_mr_then_zerr = prev_was_mr
            prev_was_mr = False

        elif _xerr_re.match(line):
            targets = line.split(") ", 1)[1]
            p = px_mcm if prev_was_mr_then_zerr else px_idle
            out_lines.append(f"X_ERROR({p}) {targets}")
            prev_was_mr = False
            prev_was_mr_then_zerr = False

        elif pm is not None and _mr_re.match(line):
            targets = line.split(") ", 1)[1]
            out_lines.append(f"MR({pm}) {targets}")
            prev_was_mr = True
            prev_was_mr_then_zerr = False

        else:
            out_lines.append(line)
            prev_was_mr = line.startswith("MR ") or line.startswith("MR(")
            prev_was_mr_then_zerr = False

    return stim.Circuit("\n".join(out_lines))


def add_mcm_depol(circuit, strength):
    """Insert DEPOLARIZE1 after every MR / Z_ERROR / X_ERROR sequence.

    When MR(...) is followed by Z_ERROR(...) then X_ERROR(...), a
    DEPOLARIZE1(strength) instruction is appended on the same targets
    as the X_ERROR.

    Parameters
    ----------
    circuit : stim.Circuit
        Input stim circuit.
    strength : float
        DEPOLARIZE1 noise parameter to insert.

    Returns
    -------
    stim.Circuit
    """
    lines = str(circuit).split("\n")
    out_lines = []
    _xerr_re = re.compile(r"^X_ERROR\([\d.e+-]+\)\s+(.+)$")

    prev_was_mr = False
    prev_was_mr_then_zerr = False

    for line in lines:
        out_lines.append(line)

        if prev_was_mr_then_zerr:
            m = _xerr_re.match(line)
            if m:
                out_lines.append(f"DEPOLARIZE1({strength}) {m.group(1)}")

        is_mr = line.startswith("MR ") or line.startswith("MR(")
        is_zerr = line.startswith("Z_ERROR(")

        prev_was_mr_then_zerr = prev_was_mr and is_zerr
        prev_was_mr = is_mr if not is_zerr else False

    return stim.Circuit("\n".join(out_lines))
