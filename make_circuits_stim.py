# Copyright IonQ Inc., all rights reserved
# =========================================================== #
# Scripts to construct generic circuit directly from matrices #
# =========================================================== #

import stim
import numpy as np
from numpy import ndarray

idle_error_type = 'Z_ERROR' # <---- replace with DEPOLARIZE1 if desired.

def stim_rotate_basis(qc0, allqb, datablock, p1, pi):
    for i in datablock:
        qc0.append("H", [i])
        qc0.append("DEPOLARIZE1", [i], p1)
        idleset = allqb.difference(set({i}))
        qc0.append(idle_error_type, idleset, pi)
        qc0.append("TICK")

def stim_reset_ancillae(qc0, allqb, anclblock, p1, pi):
    qc0.append("RX", anclblock)
    qc0.append("DEPOLARIZE1", anclblock, p1)
    idleset = allqb.difference(set(anclblock))
    qc0.append(idle_error_type, idleset, pi)
    qc0.append("TICK")

def stim_meas_ancillae(qc0, allqb, anclblock, pm, pmcm):
    qc0.append("MX", anclblock, pm)
    idleset = allqb.difference(set(anclblock))
    qc0.append(idle_error_type, idleset, pmcm)
    qc0.append("TICK")

def stim_insert_regular_dets(qc0, na, skip, with_last=False):
    for i in range(na):
        recs = [stim.target_rec(i-na)]
        if with_last:
            recs.append(stim.target_rec(i-na-skip))
        qc0.append("DETECTOR", recs)
    qc0.append("TICK")

def calculate_skip(Hx, Hz, basis='x'):
    # Check conformity with CX vs CZ ordering
    # in stim_do_syndrome_gates()
    nax, nd = Hx.shape
    naz, _  = Hz.shape

    if basis=='x':
        skip = nd+nax+naz
    elif basis=='z':
        skip = nd+naz
    else:
        raise ValueError("Invalid basis.")
    
    return skip

def stim_insert_final_dets(qc0, Hmat, skip):
    _, nd = Hmat.shape

    for i,row in enumerate(Hmat):
        indices = np.nonzero(row)[0]
        recs = [stim.target_rec(i-skip)]
        for j in indices:
            recs.append(stim.target_rec(j-nd))
        qc0.append("DETECTOR", recs)
    qc0.append("TICK")
    
def stim_insert_logicals(qc0, L):
    _, nd = L.shape

    for i,row in enumerate(L):
        indices = np.nonzero(row)[0]
        recs = [stim.target_rec(i-nd) for i in indices]
        qc0.append("OBSERVABLE_INCLUDE", recs, i)
    qc0.append("TICK")

def stim_do_syndrome_gates(qc0, Hmat, gatename, allqb, p2, pi):
    _, nd = Hmat.shape

    for i, row in enumerate(Hmat):
        indices = (np.nonzero(row)[0])
        for j in indices:
            qc0.append(gatename, [i+nd, j])
            qc0.append("DEPOLARIZE2", [i+nd, j], p2)
            idleset = allqb.difference(set({i+nd, j}))
            qc0.append(idle_error_type, idleset, pi)
            qc0.append("TICK")

def build_stim(Hx:ndarray, Hz:ndarray, L:ndarray, rounds:int,\
        p1:float, p2:float, pi:float, pm:float, pmcm:float, basis='x'):

    Hmat = {'x':Hx, 'z':Hz}[basis]

    na, nd = Hmat.shape
    datablock = list(range(nd))
    allqb = set(range(nd+na))

    qc0 = stim.Circuit()

    if basis=='x': # Rotate datablock
        stim_rotate_basis(qc0, allqb, datablock, p1, pi)

    for rnd in range(rounds):
        # Implement Hx
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+Hx.shape[0])), p1, pi)
        stim_do_syndrome_gates(qc0, Hx, "CX", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+Hx.shape[0])), pm, pmcm)
        if basis=='x': # Add regular detectors for intended basis.
            stim_insert_regular_dets(qc0, Hx.shape[0], Hx.shape[0]+Hz.shape[0], (rnd>0))
        
        # Implement Hz
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+Hz.shape[0])), p1, pi)
        stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+Hz.shape[0])), pm, pmcm)
        if basis=='z': # Add regular detectors for intended basis.
            stim_insert_regular_dets(qc0, Hz.shape[0], Hx.shape[0]+Hz.shape[0], (rnd>0))

    if basis=='x': # Rotate datablock
        stim_rotate_basis(qc0, allqb, datablock, p1, pi)
    
    # Final meas.
    qc0.append("MRZ", datablock, pm)

    # Add final detectors.
    stim_insert_final_dets(qc0, Hmat, calculate_skip(Hx, Hz, basis))

    # Add logicals
    stim_insert_logicals(qc0, L)

    return qc0

def make_cat(register:list[int], p=0):
    q_cat = stim.Circuit()
    q_cat.append("RZ", register)
    q_cat.append("H", register[0])
    qargs = []
    for i in range(len(register)-1):
        qargs.extend([register[i], register[i+1]])
    q_cat.append("CX", qargs)
    if p>0.:
        q_cat.append("DEPOLARIZE1", register, p)

    return q_cat

def do_logical_meas(circ:stim.Circuit, template:stim.Circuit,\
    datablock:list[int], catblock:list[int], logical:list[int])->None:
    logical_target = np.nonzero(logical)[0]
    qargs_cx = []
    for target, catqubit in zip(logical_target, catblock):
        qargs_cx.extend([catqubit, datablock[target]])

    # Add logical meas. gates
    circ += template
    circ.append('CY', qargs_cx)
    circ.append("H", catblock)
    circ.append("MZ", catblock)

def do_ly_prep(c:stim.Circuit, Lx:ndarray, Lz:ndarray, nd:int, obs_ind:int=-1, prev_offset:int=0):
    weights = np.sum(np.logical_or(Lx, Lz), axis=1)
    width = np.amax(weights)
    cat = list(range(nd, nd+width))

    for row_x, row_z in zip(Lx, Lz):
        j = 0
        c += make_cat(cat)
        for i in range(len(row_x)):
            if row_x[i] and row_z[i]:
                c.append("CY", [cat[j], i])
                j += 1
            elif row_x[i]:
                c.append("CX", [cat[j], i])
                j += 1
            elif row_z[i]:
                c.append("CZ", [cat[j], i])
                j += 1
        c.append("H",  cat)
        c.append("MZ", cat)
        c.append("RZ", cat)
        if obs_ind>=0:
            receivers = [stim.target_rec(i-len(cat)) for i in range(len(cat))]
            if prev_offset > 0:
                receivers += [stim.target_rec(i-prev_offset) for i in range(len(cat))]
            c.append("OBSERVABLE_INCLUDE", receivers, obs_ind+j)
        c.append("TICK")


def build_stim_y(Hx:ndarray, Hz:ndarray, Lx:ndarray, Lz:ndarray, rounds:int,\
        p1:float, p2:float, pi:float, pm:float, pmcm:float):

    nax, nd = Hx.shape
    naz, _  = Hz.shape
    na = max(nax, naz)
    allqb = set(range(nd+na))

    qc0 = stim.Circuit()

    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_do_syndrome_gates(qc0, Hx, "CX", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    k = Lx.shape[0]
    width = int(np.amax(np.sum(np.logical_or(Lx, Lz), axis=1)))

    do_ly_prep(qc0, Lx, Lz, nd)
    qc0.append("TICK")

    for rnd in range(rounds):
        # Implement Hx
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), p1, pi)
        stim_do_syndrome_gates(qc0, Hx, "CX", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), pm, pmcm)
        if rnd == 0:
            stim_insert_regular_dets(qc0, nax, nax + naz + k*width, True)
        else:
            stim_insert_regular_dets(qc0, nax, nax+naz, True)

        # Implement Hz
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), p1, pi)
        stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), pm, pmcm)
        if rnd == 0:
            stim_insert_regular_dets(qc0, naz, k*width + nax + naz, True)
        else:
            stim_insert_regular_dets(qc0, naz, nax+naz, True)

        qc0.append("TICK")
    
    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_do_syndrome_gates(qc0, Hx, "CX", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_insert_regular_dets(qc0, nax, nax+naz, True)
    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    stim_insert_regular_dets(qc0, naz, nax+naz, True)


    prev_offset = (k + 1) * width + (rounds+1) * (nax + naz)
    do_ly_prep(qc0, Lx, Lz, nd, obs_ind=0, prev_offset=prev_offset)

    return qc0

def build_stim_z(Hx:ndarray, Hz:ndarray, Lz:ndarray, rounds:int,\
        p1:float, p2:float, pi:float, pm:float, pmcm:float):

    nax, nd = Hx.shape
    naz, _  = Hz.shape
    na = max(nax, naz)
    datablock = list(range(nd))
    allqb = set(range(nd+na))

    qc0 = stim.Circuit()

    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_do_syndrome_gates(qc0, Hx, "CX", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)

    qc0.append("TICK")

    for rnd in range(rounds):
        # Implement Hx
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), p1, pi)
        stim_do_syndrome_gates(qc0, Hx, "CX", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), pm, pmcm)

        # Implement Hz
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), p1, pi)
        stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), pm, pmcm)
        stim_insert_regular_dets(qc0, naz, nax+naz, True)

        qc0.append("TICK")
    
    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    stim_insert_regular_dets(qc0, naz, naz, True)

    # Final meas.
    qc0.append("MRZ", datablock, pm)

    for o,lz in enumerate(Lz):
        idx = np.nonzero(lz)[0]
        w = len(datablock)
        recs = [stim.target_rec(i-w) for i in idx]
        qc0.append("OBSERVABLE_INCLUDE", recs, o)

    return qc0

def build_stim_x(Hx:ndarray, Hz:ndarray, Lx:ndarray, rounds:int,\
        p1:float, p2:float, pi:float, pm:float, pmcm:float):

    nax, nd = Hx.shape
    naz, _  = Hz.shape
    na = max(nax, naz)
    datablock = list(range(nd))
    allqb = set(range(nd+na))

    qc0 = stim.Circuit()
    qc0.append("RX", datablock)

    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_do_syndrome_gates(qc0, Hx, "CX", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)
    stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), 0, 0)

    qc0.append("TICK")

    for rnd in range(rounds):
        # Implement Hx
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), p1, pi)
        stim_do_syndrome_gates(qc0, Hx, "CX", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), pm, pmcm)
        stim_insert_regular_dets(qc0, nax, nax+naz, True)

        # Implement Hz
        stim_reset_ancillae(qc0, allqb, list(range(nd,nd+naz)), p1, pi)
        stim_do_syndrome_gates(qc0, Hz, "CZ", allqb, p2, pi)
        stim_meas_ancillae(qc0, allqb, list(range(nd,nd+naz)), pm, pmcm)

        qc0.append("TICK")
    
    stim_reset_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_do_syndrome_gates(qc0, Hx, "CX", allqb, 0, 0)
    stim_meas_ancillae(qc0, allqb, list(range(nd,nd+nax)), 0, 0)
    stim_insert_regular_dets(qc0, nax, nax+naz, True)

    # Final meas.
    qc0.append("MRX", datablock, pm)

    for o,lx in enumerate(Lx):
        idx = np.nonzero(lx)[0]
        w = len(datablock)
        recs = [stim.target_rec(i-w) for i in idx]
        qc0.append("OBSERVABLE_INCLUDE", recs, o)

    return qc0