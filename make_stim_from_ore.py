# Copyright IonQ Inc., all rights reserved
# Generate stim circuit with sequence and structure to match ore #

import stim
import numpy as np

NOISE_IDLE  = 'Z_ERROR'
NOISE_IDLE_MINOR = 'X_ERROR'
NOISE_2Q    = 'DEPOLARIZE2'
NOISE_1Q    = 'DEPOLARIZE1'

def do_idle(c, q, dom, p_idle):
    e = Exception("Qubit outside of allowed set.")

    if isinstance(q,int):
        if not q in dom: raise e
        idleset = dom.difference(set({q}))
    elif isinstance(q,set):
        if not q.issubset(dom): raise e
        idleset = dom.difference(q)

    c.append(NOISE_IDLE, idleset, p_idle)
    c.append(NOISE_IDLE_MINOR, idleset, p_idle/4)

def do_gpi2(c, q, p, dom, p1=0., p_idle=0.):
    if p==0.25:
        c.append("Z", [q])
        c.append("H", [q])
    elif p==0.75:
        c.append("H", [q])
        c.append("Z", [q])
    elif p==0.5:
        c.append("S", [q])
        c.append("H", [q])
        c.append("S", [q])
    elif p==0.0:
        c.append("S_DAG", [q])
        c.append("H", [q])
        c.append("S_DAG", [q])
    else:
        raise ValueError("GPI2 param invalid")
    c.append(NOISE_1Q, [q], p1)
    do_idle(c, q, dom, p_idle)

def do_gpi(c, q, p, dom, p1=0., p_idle=0.):
    if p==0.5:
        c.append("X", [q])
    elif p==0.75:
        c.append("Y", [q])
    elif p==0.25:
        c.append("Y", [q])
    elif p==0.:
        c.append("X", [q])
    else:
        raise ValueError("GPI param invalid!")
    c.append(NOISE_1Q, [q], p1)
    do_idle(c, q, dom, p_idle)

def do_zz(c, q1, q2, dom, p2=0., p_idle=0.):
    c.append("CZ", [q1, q2])
    c.append("S",  [q1, q2])

    c.append(NOISE_2Q, [q1,q2], p2)
    do_idle(c, set({q1,q2}), dom, p_idle)

def do_gz(c, q, p):
    if p==0.5:
        c.append("Z", [q])
    elif p==0.75:
        c.append("S_DAG", [q])
    elif p==0.25:
        c.append("S", [q])
    elif p==0.:
        pass
    else:
        raise ValueError("GZ param invalid!")
    
def do_ms(c, q1, q2, dom, p2=0., p_idle=0.):
    c.append("H", [q1, q2])
    c.append("CZ",[q1, q2])
    c.append("S", [q1, q2])
    c.append("H", [q1, q2])
    c.append(NOISE_2Q, [q1, q2], p2)
    do_idle(c, set({q1,q2}), dom, p_idle)
    
def do_meas(c, q, dom, pm=0., pmcm=0.):
    c.append("MRZ", q, pm)

    do_idle(c, set(q), dom, pmcm)

def do_final_meas(c, dataset, anclset, pm):
    '''Do final measurements, which are
    not explicitly instructed in ore'''
    c.append("MZ", list(anclset), pm)
    c.append("MZ", list(dataset), pm)

def ore_to_stim(ore, width, noise):
    allqb = set(range(width))
    data = set({})
    ancl = set({})

    c = stim.Circuit()
    
    for op in ore:
        g = op['gate']
        
        if g=='GPI2':
            q = op['ion']
            p = op['phase']
            do_gpi2(c, q, p, allqb, noise['p1'], noise['pi'])
        elif g=='GPI':
            q = op['ion']
            p = op['phase']
            do_gpi(c, q, p, allqb, noise['p1'], noise['pi'])
        elif g=='ZZ':
            q1 = op['ion']
            q2 = op['other']
            if op['gain']!=1:
                raise ValueError("ZZ param invalid!")
            do_zz(c, q1, q2, allqb, noise['p2'], noise['pi'])
            ancl.add(q1) # Track data 
            data.add(q2) # Track ancilla
        elif g=='MS':
            if op['with_next']:
                q1 = op['ion']
                q2 = op['other']
                do_ms(c, q1, q2, allqb, noise['p2'], noise['pi'])
                ancl.add(q1) # Track data 
                data.add(q2) # Track ancilla

        elif g=='GZ':
            q = op['ion']
            p = op['phase']
            do_gz(c, q, p)
        elif g=='MEAS':
            q = op['ions']
            do_meas(c, q, allqb, noise['pm'], noise['pmcm'])
            ancl = set({})
        else:
            raise ValueError("Bad gate name.")
    
    # If ancilla / data sets intersect, raise error
    if not data.isdisjoint(ancl):
        raise Exception("Ambiguous ancilla vs data regs")
    
    datalist = list(data)
    ancllist = list(ancl)
    datalist.sort()
    ancllist.sort()
    
    do_final_meas(c, datalist, ancllist, noise['pm'])

    return c, data, ancl

def generate_detector_indices(n_dat, n_anc, rounds, mode='xz'):
    '''Generates detector indices, for a given circuit mode
    that corresponds an experimental ordering of stabilizer checks.
    
    !!!ASSUMES!!! Equal # of x and y checks, evenly adding to n_anc.
    I.e. both X & Z parity check mats have n_anc rows.'''
    if n_anc % 2 != 0 and mode!='cheat':
        raise ValueError("n_anc must be even, by assumption.")
    
    out_dict = {}
    if mode=='xz':
        for i in range(n_anc//2):
            ix = [i+n_anc*(r-rounds)-n_dat for r in range(rounds)]
            out_dict[f"x{i}"] = ix

            iz = [i+n_anc*(r-rounds)+n_anc//2-n_dat for r in range(rounds)]
            out_dict[f"z{i}"] = iz
    elif mode=='interleaved':
        for i in range(n_anc//2):
            ix = [2*i+n_anc*(r-rounds)-n_dat for r in range(rounds)]
            out_dict[f"x{i}"] = ix

            iz = [1+2*i+n_anc*(r-rounds)-n_dat for r in range(rounds)]
            out_dict[f"z{i}"] = iz
    elif mode=='cheat':
        for i in range(n_anc):
            j = [i+n_anc*(r-rounds)-n_dat for r in range(rounds)]
            out_dict[f"x{i}"] = j
            out_dict[f"z{i}"] = j
    else:
        raise ValueError("Bad mode choice.")
    
    return out_dict

def generate_final_indices(H, basis='x'):
    '''Generate detector/logical indices for final measurements.'''
    _, n_dat = H.shape
    out_dict = {}
    for r, par in enumerate(H):
        ind = [i-n_dat for i in np.nonzero(par)[0]]
        out_dict[f"{basis}{r}"] = ind
    
    return out_dict

def add_detectors(c, H, rounds, basis='x', mode='xz'):
    n_anc_1, n_dat = H.shape
    n_anc = {'cheat':n_anc_1, 'xz':2*n_anc_1, 'interleaved':2*n_anc_1}[mode]
    mcm_dets = generate_detector_indices(n_dat, n_anc, rounds, mode)
    fin_dets = generate_final_indices(H, basis)

    for r,_ in enumerate(H):
        det_indices = mcm_dets[f"{basis}{r}"]
        fin_indices = fin_dets[f"{basis}{r}"]
        c.append("DETECTOR", stim.target_rec(det_indices[0]))
        for i in range(len(det_indices)-1):
            targs = [stim.target_rec(d) for d in det_indices[i:i+2]]
            c.append("DETECTOR", targs)

        targs = [stim.target_rec(d) for d in fin_indices]\
            + [stim.target_rec(det_indices[-1])]
        c.append("DETECTOR", targs)

def add_logicals(c, L):
    fin_logs = generate_final_indices(L, 'l')
    for l,indices in enumerate(fin_logs.values()):
        targs = [stim.target_rec(d) for d in indices]
        c.append("OBSERVABLE_INCLUDE", targs, l)

# >>>>> Wrapper func build a full stim circuit with decorations <<<<< #
def ore_to_stim_full(ore, n_physical_anc, H, L, rounds, basis, mode, noise):
    n_anc, n_dat = H.shape
    width = n_dat + n_physical_anc

    c, _, _ = ore_to_stim(ore, width, noise)
    
    n_meas = c.num_measurements
    if mode=='cheat':
        n_bits = n_anc*rounds + n_dat
    else:
        n_bits = 2*n_anc*rounds + n_dat
    if n_bits != n_meas:
        # !!!ASSUMES!!! Equal # of x and y checks,
        # I.e. n_anc is # of rows for H in either basis.
        raise Exception("Mismatched number of bits.")

    add_detectors(c, H, rounds, basis, mode)
    add_logicals(c, L)

    return c