# Copyright IonQ Inc., all rights reserved
import numpy as np
from pickle import load, dump
import csv

def get_qubit_map(mongo_record, data_len, anc_len):
    qubitmap = mongo_record['info']['params']['qubitMap']

    return qubitmap[0:data_len], qubitmap[data_len:data_len+anc_len]

def consolidate_records(data_dictionary):
    '''Consolidate shots from many datasets
    Input:
        Dictionary whose keys are prognames identifying an ore file.
        Its values are lists whose entries correspond to a distinct experimental
        run. Each list entry is a tuple returned by get_raw_data().'''
    consolidated_data = {}

    for circ in data_dictionary:
        datasets = data_dictionary[circ]

        # Total shots across consolidated datasets
        nshots = np.sum([raw_data[0].shape[0] for raw_data in datasets])
        n = (nshots,)

        # Get other dimensions
        dat0 = datasets[0]
        dat = np.zeros(n+dat0[0].shape[1:], dtype=np.bool_)
        mcm = np.zeros(n+dat0[1].shape[1:], dtype=np.bool_)
        
        # Consolidate
        i = 0
        for raw_data in datasets:
            d, m = raw_data
            shots = d.shape[0]
            dat[i:i+shots] = d[:]
            mcm[i:i+shots] = m[:]

            i += shots
        
        consolidated_data[circ] = (dat, mcm)
    
    return consolidated_data

def build_ion_drb_lut(fname):
    '''Read in spanning DRB in file.
    Build a lookup dictionary indexed by ion pairs.'''
    with open(fname) as f:
        data = {tuple(eval(row['pair'])): row['infidelity'] for row in csv.DictReader(f)}

    drb = {} # Build LUT, drb[(ion0, ion1)] = infidelity
    for row in data:
        infidelity = float(data[row]) / 1e4
        q0 = min(row)
        q1 = max(row)
        drb[(q0,q1)] = infidelity
        drb[(q1,q0)] = infidelity

    return drb

def read_qubit_map(fname):
    handle = open(fname, 'rb')
    qbmap = load(handle)
    handle.close()

    return qbmap

def build_qubit_drb_map(iondrb, data, ancillae):
    '''Given lists of ions comprising data and ancillae,
    build a lookup table from iondrb, with tuples of qubits
    as keys instead of ions.'''
    qbdrb = {}
    for i,d in enumerate(data): # i,j are qubits
        for j,a in enumerate(ancillae): # d, a are ions
            k = j + len(data) # Ancillae shifted to start AFTER data qubit indices
            infidelity = iondrb[(d,a)]
            q0 = min(i,k)
            q1 = max(i,k)
            qbdrb[(q0,q1)] = infidelity
            qbdrb[(q1,q0)] = infidelity
    
    return qbdrb

def write_consolidated_records(data_dictionary, fnames=['']):
    '''Expects that there are as many strings in fname as
    there are top-level keys in data_dictionary.
    Saves each top-level entry of data_dictionary into
    three numpy text files, with prefixes in fname.'''
    for key,name in zip(data_dictionary, fnames):
        d, m = data_dictionary[key]
        np.save(f"{name}_d", d, allow_pickle=False)
        np.save(f"{name}_m", m, allow_pickle=False)

def read_consolidated_records(fnames=['']):
    '''Expects that there are as many strings in fname as
    there are top-level keys in data_dictionary.
    Saves each top-level entry of data_dictionary into
    three numpy text files, with prefixes in fname.'''
    data_dictionary = {}
    for name in fnames:
        d = np.load(f"{name}_d.npy", allow_pickle=False)
        m = np.load(f"{name}_m.npy", allow_pickle=False)
        data_dictionary[name] = (d, m)
    
    return data_dictionary

def apply_mask(bits, mask):
    '''If MCM or final bits are deterministically flipped,
    say by DD, apply a mask to unflip them.'''
    for r,shot in enumerate(bits):
        bits[r] = np.logical_xor(shot, mask)

def trim_final_anc(final_anc, bits_to_keep):
    # final_anc should be [shots, bits]
    return np.take(final_anc, bits_to_keep, axis=1)

def get_raw_data(data_record, mcmbits:int, datablock:list[int], anclblock:list[int], filterSSL=False):
    '''Given logfile name, and a MongoDB collection, return raw counts, split into:
    Returns:f
        meas_data: Final measurement, data register [shot, qubit].
        meas_ancl: Final measurement, ancilla register [shot, qubit].
        mcm_ancl:  MCM records [shot, MCM round, qubit]'''
    meas = np.array(data_record['results']['raw']['thresholdUnfiltered'])
    chck_final = np.array(data_record['results']['raw']['thresholdSSL'])
    # chck1 = np.any(np.take(chck_final, datablock, axis=1), axis=1) # True if bad
    chck1 = np.any(chck_final, axis=1) # True if bad
    nshots = len(meas)

    meas_data = np.take(meas, datablock, axis=1)
    meas_ancl = np.take(meas, anclblock, axis=1)

    if mcmbits<=len(anclblock):
        mcm_ancl = meas_ancl[:,0:mcmbits]
        if filterSSL:
            isPass = np.logical_not(chck1)
            meas_data = meas_data[isPass, :]
            mcm_ancl = mcm_ancl[isPass, :]
    else:
        rawmcm = np.array(data_record['results']['raw']['thresholdMCM'])
        chck0 = np.take(# SSL check bits
            np.take(rawmcm, np.arange(1, len(rawmcm), 2), axis=0),
            anclblock, axis=1
        )
        mcmv = np.take(rawmcm, np.arange(0, len(rawmcm), 2), axis=0) # Raw MCM values

        # Index: [shot #, MCM round, ion]
        mcm_by_ancillae = np.take(mcmv, anclblock, axis=1) # MCM, sliced by ancilla
        mcm_ancl = np.zeros((nshots, mcmbits), dtype=np.bool_)
        mcm_chck = np.zeros(nshots, dtype=np.bool_)
        s = 0 # Shot index
        b = 0 # MCM bit index
        l = len(anclblock) # Num. of physical ancillae
        for row_anc, row_chck in zip(mcm_by_ancillae, chck0):
            bmax = min(b+l, mcmbits)
            rmax = bmax - b
            mcm_ancl[s,b:bmax] = row_anc[0:rmax]
            mcm_chck[s] = (not np.all(row_chck)) or mcm_chck[s] # True if bad
            if bmax+l>=mcmbits:
                mcm_ancl[s,bmax:] = meas_ancl[s,0:mcmbits-bmax]
                b = 0
                s += 1
            else: b = bmax
        
        if s<nshots: raise Exception("Underfilled ancilla bits from MCM")
        if b>0: raise Exception("Underfilled ancilla bits from MCM")
        # If overfilled, numpy will raise an error.

        if filterSSL:
            isPass = np.logical_and(np.logical_not(mcm_chck), np.logical_not(chck1))
            meas_data = meas_data[isPass, :]
            mcm_ancl = mcm_ancl[isPass, :]
    
    return meas_data, mcm_ancl

def filter_SSL(data_record, raw_data):
    meas_data, meas_ancl, mcm_ancl = raw_data

    '''Filter shots based on whether SSL detection was triggered.'''
    filter = np.any(data_record['results']['raw']['thresholdSSL'], axis=1)
    mask = np.nonzero(filter==False)[0]
    
    filtered_data = np.take(meas_data, mask, axis=0)
    filtered_ancl = np.take(meas_ancl, mask, axis=0)
    filtered_mcm  = np.take(mcm_ancl,  mask, axis=0)

    return filtered_data, filtered_ancl, filtered_mcm

def assemble_ancillae(raw_data, n_checks, basis='x', mode='xz'):
    '''In circuits where ancillae are sometimes measured via MCM and
    at other times via final projective meas., combine them into one
    coherent matrix, still indexed the same way as in get_raw_data().
    
    !!!ASSUMPTION!!!: Ancillae ions are optimally packed. I.e.,
    (Rounds of MCM x Ancilla ions) = (Logical rounds x No. of checks)
    
    !!!ASSUMPTION!!!: In mode=="interleaved", assume there are equal
    numbers of X and Z checks. This is true of BB and GB codes we use.'''
    databits, mcm_anc = raw_data
    
    nax, naz = n_checks # of checks in X or Z bases.
    nshots = databits.shape[0]

    if mcm_anc.shape[0] != nshots: # Check if inputs have identical # of shots.
        raise Exception("Final and MCM ancillae arrays have inconsistent # of shots.")
    
    ancillae = mcm_anc.reshape(nshots, -1)

    if ancillae.shape[1] % (nax+naz) != 0:
        raise Exception("# of ancillae bits incongruent with # of checks.")
    
    n_logical_cycles = ancillae.shape[1] // (nax+naz)
    
    if mode=='xz':
        bitblock = {'x':(0,nax),'z':(nax,nax+naz)}[basis]
        ancillae_by_round = ancillae.reshape(nshots, n_logical_cycles, -1)
        return ancillae_by_round[:, :, bitblock[0]:bitblock[1]]
    
    elif mode=='interleaved':
        bitblock = {'x':0, 'z':1}[basis]
        ancillae_by_round = ancillae.reshape(nshots, n_logical_cycles, nax, 2)
        return ancillae_by_round[:, :, :, bitblock]

    elif mode=='cheat':
        bitblock = {'x':(0,nax),'z':(0,naz)}[basis]
        ancillae_by_round = ancillae.reshape(nshots, n_logical_cycles, -1)
        return ancillae_by_round[:, :, bitblock[0]:bitblock[1]]
    
    else:
        raise ValueError("Bad mode choice")
    

def get_data_parity(raw_data, paritymat):
    final_dat, _ = raw_data
    '''Compute detector/logical values from final measurement data'''
    if final_dat.shape[1] != paritymat.shape[1]:
        raise Exception("Incompatible # of data qubits.")

    nshots, nqubits = final_dat.shape

    det_temp = final_dat.reshape(nshots, 1, nqubits) * paritymat.reshape(1, -1, nqubits)
    det_vals = np.mod(np.sum(det_temp, axis=2), 2)

    return np.array(det_vals, dtype=np.bool_)

def add_data_syndrome(syndrome_bits, raw_data, Hmat):
    '''Append parities computed from data qubits to ancilla bits,
    as if they are another round of syndromes.'''
    data_syndrome = get_data_parity(raw_data, Hmat)
    nshots = syndrome_bits.shape[0]
    nsyndromebits = syndrome_bits.shape[2]
    
    if data_syndrome.shape[0] != nshots:
        raise Exception("Mismatched # of shots")
    
    if data_syndrome.shape[1] != nsyndromebits:
        raise Exception("Mismatched # of syndrome bits in 1 round")
    
    return np.concatenate((
        syndrome_bits,
        data_syndrome.reshape(nshots, 1, nsyndromebits)
    ), axis=1)

def get_detector_values(syndromes):
    '''Extract detector values given syndrome data.
    Specifically, layered detectors compare syndrome data from
    consecutive rounds and return 1 iff they disagree.'''
    _, nrounds, _ = syndromes.shape
    
    detector_values = np.zeros(syndromes.shape, dtype=np.bool_)

    detector_values[:,0,:] = syndromes[:,0,:]
    for r in range(1, nrounds):
        detector_values[:,r,:] = np.logical_xor(
            syndromes[:, r-1, :],
            syndromes[:, r  , :]
        )
    
    return detector_values

def write_decoding_dict(decdict, fname):
    '''Write decoding dictionary to file.
    Assumes decode_dict is keyed by circuitname, and whose
    values are 4-tuples: (Decoder, Detector bits, Logical bits, Circuit)'''
    bits = {c:(decdict[c][1],decdict[c][2],decdict[c][3]) for c in decdict.keys()}
    handle = open(fname, 'wb')
    dump(bits, handle)
    handle.close()
    del handle

def read_decoding_dict(fname, decoder_init):
    handle = open(fname, 'rb')
    bits = load(handle)
    handle.close()
    del handle

    decoding_dict = {}
    for c in bits.keys():
        dem = bits[c][2].detector_error_model()
        dec = decoder_init(dem)
        decoding_dict[c] = (dec, bits[c][0], bits[c][1], bits[c][2])

    return decoding_dict

# >>>>> Wrapper for above funcs; extract all relevant bits for decoding <<<<<< #
def extract_all_decoding_bits(raw_data, Hx, Hz, L, basis='x', mode='xz', lval=0):
    '''Convenience wrapper for functions above.
    Extract all logical bit values and detector flip sets;
    return them in a format suitable for plugging into decoder.'''
    nax, _ = Hx.shape
    naz, _ = Hz.shape
    H = {'x':Hx,'z':Hz}[basis]

    # Extract detector flip sets.
    bits_anc = assemble_ancillae(raw_data, (nax, naz), basis, mode)
    bits_syn = add_data_syndrome(bits_anc, raw_data, H)
    bits_det = get_detector_values(bits_syn)

    # Extract logicals
    bits_log = get_data_parity(raw_data, L)
    if lval==1:
        bits_log = np.logical_not(bits_log)

    return bits_det, bits_log
# >>>>> =============================================================== <<<<<< #

def decode_one(decoder, detectorflips, logicalvalues):
    '''Decode a single shot.'''
    raw_has_fault = np.any(logicalvalues) # Raw logical value has fault(s)

    # Decode (!!!HACK: Transposed detectorflips to accord with new ore-to-stim detector convention)
    prediction = decoder.decode(detectorflips.T.flatten())
    
    # If prediction DON'T match experimental logical value, count a logical fault
    is_correct = (prediction==logicalvalues)
    dec_has_fault = (not np.all(is_correct)) # Decoded value has fault(s)

    return raw_has_fault, dec_has_fault, is_correct

def get_logical_perf(decoder, detectorflips, logicalvalues):
    '''Wrapper script to get logical performance,
    sequentially aggregating one shot at a time.
    
    For many shots, consider instead pipefunc on decode_one().'''

    rawstat = decstat = shots = 0
    for arg in zip(decoder, detectorflips, logicalvalues):
        raw, dec = decode_one(*arg)
        
        shots += 1
        rawstat += raw
        decstat += dec
    
    return rawstat, decstat, shots