import numpy as np
import compilescripts as comp

def ore_syndrome_round(qc, H, offset, basis='x', native_2q='ZZ'):
    if native_2q=='ZZ':
        gen_code = {'x':comp.gen_x_checks, 'z':comp.gen_z_checks}[basis]
    elif native_2q=='XX':
        gen_code = lambda x,y: comp.gen_checks_XX(x,y,basis)
    _, nd = H.shape
    for i, row in enumerate(H):
        indices = [int(q) for q in np.nonzero(row)[0]]
        subcirc = gen_code(i+nd+offset, indices)

        qc.extend(subcirc)

def build_ore(Hx, Hz, rounds, n_ancilla, basis='x', do_echo=False):
    nax, nd = Hx.shape
    naz, _  = Hz.shape
    nbits = (nax+naz)*rounds
    qc = []

    if basis=='x':
        subcirc = comp.gen_h(list(range(nd)))
        qc.extend(subcirc)
    
    round = 0
    bits = 0
    used_anc = 0
    while round<rounds:
        i_Hx = 0
        i_Hz = 0
        isX = (Hx.shape[0]>0)
        while i_Hx+i_Hz<nax+naz:
            if isX and len(Hx)>0:
                Hsub = Hx[i_Hx:i_Hx+1]
                ore_syndrome_round(qc, Hsub, used_anc, 'x')
                i_Hx += 1
                used_anc += 1
                if i_Hx>=len(Hx) and len(Hz)>0:
                    isX = False
            
            elif (not isX) and len(Hz)>0:
                Hsub = Hz[i_Hz:i_Hz+1]
                ore_syndrome_round(qc, Hsub, used_anc, 'z')
                i_Hz += 1
                used_anc += 1
                if i_Hz>=len(Hz) and len(Hx)>0:
                    isX = True

            if (bits<nbits-n_ancilla) and (used_anc==n_ancilla):
                used_anc = 0
                subcirc = comp.gen_mcm(list(range(nd, nd+n_ancilla)))
                bits += n_ancilla
                qc.extend(subcirc)
                if do_echo:
                    qc.extend(comp.gen_echo(list(range(nd))))

        round += 1

    if basis=='x':
        subcirc = comp.gen_h(list(range(nd)))
        qc.extend(subcirc)
    
    return qc

def build_ore_interleaved(Hx, Hz, rounds, n_ancilla, basis='x', do_echo=False, native_2q='ZZ', initvals=[]):
    nax, nd = Hx.shape
    naz, _  = Hz.shape
    nbits = (nax+naz)*rounds
    qc = []

    if basis=='x':
        if len(initvals)==0:
            for q in range(nd):
                qc.append(comp.syntax_gpi2(q, 0.25))
        else:
            for q in range(nd):
                if initvals[q]:
                    qc.append(comp.syntax_gpi2(q, 0.75))
                else:
                    qc.append(comp.syntax_gpi2(q, 0.25))
    
    if basis=='z' and len(initvals)>0:
        for q in range(nd):
            if initvals[q]:
                qc.append(comp.syntax_gpi(q, 0.))
    
    round = 0
    bits = 0
    used_anc = 0
    while round<rounds:
        i_Hx = 0
        i_Hz = 0
        isX = True
        while i_Hx+i_Hz<nax+naz:
            if isX:
                Hsub = Hx[i_Hx:i_Hx+1]
                ore_syndrome_round(qc, Hsub, used_anc, 'x', native_2q)
                if i_Hz<len(Hz):
                    isX = not isX
                i_Hx += 1
                used_anc += 1
            
            else:
                Hsub = Hz[i_Hz:i_Hz+1]
                ore_syndrome_round(qc, Hsub, used_anc, 'z', native_2q)
                if i_Hx<len(Hx):
                    isX = not isX
                i_Hz += 1
                used_anc += 1

            if (bits<nbits-n_ancilla) and (used_anc==n_ancilla):
                used_anc = 0
                subcirc = comp.gen_mcm(list(range(nd, nd+n_ancilla)))
                bits += n_ancilla
                qc.extend(subcirc)
                if do_echo:
                    qc.extend(comp.gen_echo(list(range(nd))))

        round += 1

    if basis=='x':
        for q in range(nd):
            qc.append(comp.syntax_gpi2(q, 0.75))
    
    return qc

def build_ore_cheat(Hx, Hz, rounds, n_ancilla, basis='x'):
    if basis=='x':
        Hz_null = np.array([[]], dtype=np.bool_).reshape(0,Hz.shape[1])
        qc = build_ore(Hx, Hz_null, rounds, n_ancilla, basis='x', do_echo=False)
    elif basis=='z':
        Hx_null = np.array([[]], dtype=np.bool_).reshape(0,Hx.shape[1])
        qc = build_ore(Hx_null, Hz, rounds, n_ancilla, basis='z', do_echo=False)
    else:
        qc = None
        raise ValueError("Bad Basis.")
    
    return qc