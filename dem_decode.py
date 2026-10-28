import analysis
# import make_circuits_stim
from stim import Circuit
from compilescripts import read_from_file
import make_stim_from_ore as o2s
import numpy as np
import stimbposd
from Beam import myBPListV3
import pymatching

import BB5_def, GB4_def, extra_codes_def

default_noise_model = {
    'p1': 8e-4,
    'p2': 1.75e-2,
    'pi': 5e-3/32,
    'pm': 9e-3,
    'pmcm': 1.392e-4
}

default_bposd_kwargs = {
    "max_bp_iters":10_000,
    "bp_method":"min_sum",
    "osd_order":5,
    "osd_method":"osd_cs"
}

default_bplist_kwargs = { # Beam32 preset
    'max_rounds':10,
    "list_size":32,
    "initial_iters":40,
    "iters_per_round":30
}

# default_bplist_kwargs = { # Beam64 preset
#     'max_rounds':20,
#     "list_size":64,
#     "initial_iters":40,
#     "iters_per_round":30
# }

def decoder_constructor(dem, decname='beam', kwargs=default_bplist_kwargs):
    if decname=='bposd':
        dec = stimbposd.BPOSD(dem, **kwargs)
    elif decname=='beam':
        dec = myBPListV3(dem, **kwargs)
    elif decname=='pymatching':
        dec = pymatching.Matching.from_detector_error_model(dem)
    else: raise ValueError("Bad decoder name.")

    return dec

def init_decoder(orename, Hx, Hz, L, rounds, basis, n_phy_anc, mode, noise, decname='beam'):
    ore = read_from_file(orename)
    H = {'x':Hx, 'z':Hz}[basis]
    circ = o2s.ore_to_stim_full(ore, n_phy_anc, H, L, rounds, basis, mode, noise)
    dem = circ.detector_error_model()

    dec_kwargs = {'beam':default_bplist_kwargs, 'bposd':default_bposd_kwargs}[decname]
    dec = decoder_constructor(dem, decname=decname, kwargs=dec_kwargs)

    return dec, circ

def parse_fname(fname):
    repstring = fname.strip('ore/').strip('.conf.js')
    repstring = repstring.replace('FourTwoConcat_','422_')
    repstring = repstring.replace('Toric_', 'Tor_')
    fstr, dstr, rstr, bstr = repstring.split('_')
    code_fam = fstr[0:3]
    variants = fstr[3: ]
    rounds = int(rstr[1:])

    is_cheat, is_row_reduced, mode = {
        'reduced'   :(False, True , 'xz'),
        'redint'    :(False, True , 'interleaved'),
        'redintXX'  :(False, True , 'interleaved'),
        'DD'        :(False, False, 'xz'),
        'cheat'     :(True , True,  'cheat'),
        'final'     :(False, True , 'interleaved'),
        ''          :(False, True , 'interleaved') # Changed to default to interleaved since final data on Feb12.
    }[variants]

    dist  = int(dstr[1:])
    basis = bstr[0].lower()
    lval  = int(bstr[1])

    # !!!ASSUMES!!! Particular families defined in _def.py
    code_param = {
        'GB4': {4: (16,2,4), 5: (26,2,5)},
        'BB5': {3: (18,4,3), 4: (24,4,4), 5: (30,4,5)},
        'Tor': {3: (18,2,3), 4: (16,2,4)},
        '422': {2: (4,2,2) , 4: (16,4,4)}
    }[code_fam][dist]

    # !!!ASSUMES!!! Ancilla ion config for experiment in CircuitGen
    n_a_physical = {
        False: { # Non cheat 
            'GB4': {
                False: {4: 16},
                True : {4: 14, 5: 12},
            },
            'BB5': {
                False: {3: 18, 4: 12, 5: 10},
                True : {3: 14, 4: 10, 5: 10}
            },
            'Tor': {
                False: {3: 16, 4: 14},
                True : {3: 16, 4: 14}
            },
            '422': {
                False: {2: 2, 4: 12},
                True : {2: 2, 4: 12}
            }
        },
        True: { # Cheat 
            'GB4': {
                True : {4: 7, 5: 12},
            },
            'BB5': {
                True : {3: 7, 4: 10, 5: 10}
            }
        }
    }[is_cheat][code_fam][is_row_reduced][dist]

    return code_fam, code_param, n_a_physical, basis, is_row_reduced, mode, rounds, lval

def get_matrices(code_fam, code_param, is_row_reduced, basis):
    definitions_file = {'GB4':GB4_def, 'BB5':BB5_def, 'Tor':extra_codes_def, '422':extra_codes_def}[code_fam]
    if is_row_reduced:
        matrix_definitions = definitions_file.definitions_row_reduced
    else:
        matrix_definitions = definitions_file.definitions
    
    # HACK. If code is defined in extra defs, keys are custom strings, not code params.
    if code_fam=='422':
        code_param = {(16,4,4):'(16,4,4)-422concat',(4,2,2):'(4,2,2)-unconcat'}[code_param]
    elif code_fam=='Tor':
        code_param = {
            (16,2,4):'(16,2,4)-checkerboard',
            (18,2,3):'(18,2,3)-toric'}[code_param]

    logical_key = basis.upper() + "-logical"
    matrices = matrix_definitions[code_param]

    return matrices['X-parity'], matrices['Z-parity'], matrices[logical_key]

def get_code_objects(code_fam, code_param, is_row_reduced, basis, ore, n_phy_anc, rounds, mode, noise={}):
    '''For a given BB / GB code, collect objects relevant to defining/using
    the code, including parity and logical matrices, and decoder object.'''

    Hx, Hz, L = \
        get_matrices(code_fam, code_param, is_row_reduced, basis)
    
    if len(noise.keys())==0:
        nm = default_noise_model
    else:
        nm = noise

    dec, circ = init_decoder(ore, Hx, Hz, L, rounds, basis, n_phy_anc, mode, nm)

    return dec, Hx, Hz, L, circ

def adjust_stim_circuit_noise(circ, name, p2new):
    '''Ad-hoc adjustment of 2q error rate in Stim DEM for specific bad pairs'''
    pairdict = { # "Bad" pairs from DRB
        "BB5final_d3": (18, 2),
        "BB5final_d4": (24, 2),
        "BB5final_d5": (30, 2),
        "FourTwoConcat_d2": (4 , 2),
        "FourTwoConcat_d4": (16, 2),
        "Toric_d3": (18, 2),
        "Toric_d4": (16, 2)
    }
    # p2default = 1.25e-2

    nameinclusion = [(n in name) for n in pairdict.keys()]
    if not np.any(nameinclusion):
        # pair = (-1,-1)
        return circ
    else:
        keyindex = nameinclusion.index(True)
        key = list(pairdict.keys())[keyindex]
        pair = pairdict[key]

        out = Circuit()
        i_start = 0
        for i_curr,op in enumerate(circ):
            isPair = False
            if (op.name=='DEPOLARIZE2'):
                qargs = [q.value for q in op.target_groups()[0]]
                isPair = (qargs[0]==pair[0]) and (qargs[1]==pair[1])
            if isPair:
                if i_curr>i_start: out += circ[i_start:i_curr]
                i_start = i_curr + 1
                out.append("DEPOLARIZE2", [pair[0], pair[1]], p2new)
            # elif (op.name=='DEPOLARIZE2'):
            #     if i_curr>i_start: out += circ[i_start:i_curr]
            #     i_start = i_curr + 1
            #     qargs = [q.value for q in op.target_groups()[0]]
            #     out.append("DEPOLARIZE2", [qargs[0], qargs[1]], p2default)

        if i_curr>i_start: out += circ[i_start:i_curr+1]

    return out


# >>>>>> Convenience funcs to extract and consolidate many datasets <<<<<< #
def extract_records(metadatalist, filterSSL=False):
    '''For each experimental metadata, grab and assemble raw data arrays'''
    _, coll = analysis.get_mongo_db()

    data = {}
    qbmap = {}
    for run in metadatalist:
        objID, progname, _ = run
        shortname = progname.split("/")[-1].replace(".conf.js","")
        
        _, code_param, n_a_physical, _, is_row_reduced, mode, rounds, _ = \
            parse_fname(progname)
        n,k,d = code_param

        # !!!ASSUMES!!! Equal numbers of X & Z checks.
        n_a_logical = {False:n, True:n-k}[is_row_reduced]
        if mode=='cheat':
            n_a_logical = n_a_logical//2
        
        rec = analysis.retrieve_record(coll, objID)
        datblk, ancblk = analysis.get_qubit_map(rec, n, n_a_physical)

        # !!!ASSUMES!!! Experimental circuit has d syndrome rounds.
        # mcm_per_log = n_a_logical//n_a_physical + ((n_a_logical%n_a_physical)>0)*1 <---- Old convention.
        # mcm_rounds = (d-1)*mcm_per_log + mcm_per_log-1 # <---- Allows unused ancillae between rounds.
        n_mcm_bits = rounds * n_a_logical

        raw_data = analysis.get_raw_data(rec, n_mcm_bits, datblk, ancblk, filterSSL)

        if shortname in data.keys():
            data[shortname].append(raw_data)
        else:
            data[shortname] = [raw_data]
            qbmap[shortname] = (datblk, ancblk)
    
    return data, qbmap

def get_decoding_bits_cyclic_from_file(fnames:list[str], noise={}):
    '''Generate necessary bits for decoding, based on data filenames.
    Expects filenames to follow convention in extract_records().'''
    if len(noise.keys())==0:
        nm = default_noise_model
    else:
        nm = noise

    data_dictionary = analysis.read_consolidated_records(fnames)

    decoding_dict = {}
    for fname in fnames:
        basename = fname.split('/')[-1]
        pathname = fname.replace(basename, '')
        code_fam, code_param, n_phy_anc, basis, is_row_reduced, mode, rounds, lval\
            = parse_fname(basename)

        orename = pathname + '../ore/' + basename + ".conf.js"
        dec, Hx, Hz, L, circ =\
            get_code_objects(code_fam, code_param, is_row_reduced, basis, orename, n_phy_anc, rounds, mode, nm)
        
        if mode=='cheat':
            if   basis=='x': Hz = np.zeros((0,Hz.shape[1]), dtype=np.bool_)
            elif basis=='z': Hx = np.zeros((0,Hx.shape[1]), dtype=np.bool_)
            else: raise Exception("Bad basis.")
        
        raw_data = data_dictionary[fname]

        detectors, logicals = \
            analysis.extract_all_decoding_bits(raw_data, Hx, Hz, L, basis, mode, lval)
        
        decoding_dict[fname] = dec, detectors, logicals, circ
    
    return decoding_dict