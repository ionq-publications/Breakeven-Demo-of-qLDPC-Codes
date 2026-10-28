# Copyright IonQ Inc., all rights reserved
from qiskit import circuit as qkc
import stim
import compilescripts as comp

def make_repcode_qiskit(dist:int, flip_data:bool=False):
    qr0 = qkc.QuantumRegister(dist + dist-1)
    cr0 = qkc.ClassicalRegister((dist-1)*(dist-1) + (dist+dist-1))
    qc_rep = qkc.QuantumCircuit(qr0, cr0)
    i_meas = 0

    # Data qubit flip
    if flip_data:
        for q in range(dist):
            qc_rep.x(q)

    for i in range(dist):
        # Ancilla prep
        for q in range(dist-1):
            qc_rep.h(q+dist)
        qc_rep.barrier()
        
        # Syndromes
        for q in range(dist-1):
            qc_rep.h(q+0)
            qc_rep.cx(q+dist, q+0)
            qc_rep.h(q+0)

            qc_rep.h(q+1)
            qc_rep.cx(q+dist, q+1)
            qc_rep.h(q+1)
        
        # Ancilla rotate
        for q in range(dist-1):
            qc_rep.h(q+dist)
        qc_rep.barrier()

        # Ancilla measure / reset
        if i<dist-1:
            for q in range(dist-1):
                qc_rep.measure(q+dist, i_meas)
                i_meas += 1
    
    # Measure all
    for q in range(dist+dist-1):
        qc_rep.measure(qr0[q], cr0[q+i_meas])

    return qc_rep

def make_repcode_qiskit_x(dist:int):
    qr0 = qkc.QuantumRegister(dist + dist-1)
    cr0 = qkc.ClassicalRegister((dist-1)*(dist-1) + (dist+dist-1))
    qc_rep = qkc.QuantumCircuit(qr0, cr0)
    i_meas = 0

    # Data qubit flip
    for q in range(dist):
        qc_rep.h(q)

    for i in range(dist):
        # Ancilla prep
        for q in range(dist-1):
            qc_rep.h(q+dist)
        # qc_rep.barrier()
        
        # Syndromes
        for q in range(dist-1):
            qc_rep.cx(q+dist, q+0)

            qc_rep.cx(q+dist, q+1)
        
        # Ancilla rotate
        for q in range(dist-1):
            qc_rep.h(q+dist)
        # qc_rep.barrier()

        # Ancilla measure / reset
        if i<dist-1:
            for q in range(dist-1):
                qc_rep.measure(q+dist, i_meas)
                i_meas += 1
    
    # Data qubit flip
    for q in range(dist):
        qc_rep.h(q)

    # Measure all
    for q in range(dist+dist-1):
        qc_rep.measure(qr0[q], cr0[q+i_meas])

    return qc_rep

def make_repcode_x_ore(dist:int):
    ore = comp.gen_h([q for q in range(dist)])

    for round in range(dist):
        for check in range(dist-1):
            ore_one_check = comp.gen_x_checks(check+dist, [check, check+1])
            ore.extend(ore_one_check)
        
        if round<dist-1:
            ore_one_mcm = comp.gen_mcm([q+dist for q in range(dist-1)])
            ore.extend(ore_one_mcm)

    ore.extend(comp.gen_h([q for q in range(dist)]))

    return ore

def make_repcode_stim(dist:int, p1:float, p2:float,\
        pi:float, pm:float, pmcm:float, flip_data:bool=False):
    qc0 = stim.Circuit()
    allqb = set({i for i in range(dist + dist - 1)})

    # Data qubit flip
    if flip_data:
        qargs = [q for q in range(dist)]
        qc0.append("X", qargs)
        qc0.append("TICK")

    for i in range(dist):
        # Ancilla prep
        for q in range(dist-1):
            qc0.append("H", [q+dist])
            qc0.append("DEPOLARIZE1", [q+dist], p1)
            idleset = allqb.difference({q+dist})
            qc0.append("DEPOLARIZE1", idleset, pi/10)
            qc0.append("TICK")
        
        # Syndromes
        for q in range(dist-1):
            qc0.append("CZ", [q+dist, q+0])
            qc0.append("DEPOLARIZE2", [q+dist, q+0], p2)
            idleset = allqb.difference({q+dist, q+0})
            qc0.append("DEPOLARIZE1", idleset, pi)
            qc0.append("TICK")
            
            qc0.append("CZ", [q+dist, q+1])
            qc0.append("DEPOLARIZE2", [q+dist, q+1], p2)
            idleset = allqb.difference({q+dist, q+1})
            qc0.append("DEPOLARIZE1", idleset, pi)
            qc0.append("TICK")
        
        # Ancilla rotate
        for q in range(dist-1):
            qc0.append("H", [q+dist])
            qc0.append("DEPOLARIZE1", [q+dist], p1)
            idleset = allqb.difference({q+dist})
            qc0.append("DEPOLARIZE1", idleset, pi/10)
            qc0.append("TICK")

        # Ancilla measure / reset
        # if i<dist-1:
        qargs = [q+dist for q in range(dist-1)]
        qc0.append("MRZ", qargs, pm)
        idleset = allqb.difference(set(qargs))
        qc0.append("DEPOLARIZE1", idleset, pi)
    
        # Add detectors
        for q in range(dist-1):
            recs = [stim.target_rec(q-dist+1)]
            if i>0:
                recs.append(stim.target_rec(q-2*dist+2))
            qc0.append("DETECTOR", recs)
        
        if i<dist-1:
            qc0.append("DEPOLARIZE1", list(range(dist)), pmcm)

    qc0.append("MZ", [q for q in range(dist)], pm)
    # Add final detectors
    for q in range(dist-1):
        recs = [stim.target_rec(q+i-dist) for i in range(2)]
        recs.append(stim.target_rec(q-dist-dist+1))
        qc0.append("DETECTOR", recs)

    # Add logicals
    n_ones = dist - ((dist-1)%2)
    qc0.append("OBSERVABLE_INCLUDE",\
        [stim.target_rec(q-dist) for q in range(n_ones)], 0)
    
    return qc0