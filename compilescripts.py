# Copyright IonQ Inc., all rights reserved
import json
import qiskit.circuit as qkc

def rename_M(indata):
    outdata = []
    for inst in indata: # Fix arguments
        if inst['gate']=='M':
            inst['gate'] = 'MEAS'
            inst.pop('c_reg')
        outdata.append(inst)
    return outdata
    
def consolidate_MEAS(indata, do_last_layer=False):
    outdata = []
    args = []
    for i,inst in enumerate(indata):
        if inst['gate']=='MEAS':
            args.append(inst['ion'])
        else:
            if len(args)>0:
                outdata.append({
                    'gate':'MEAS',
                    'ions': args
                })
                args = []
            outdata.append(inst)

    if do_last_layer and len(args)>0:
        outdata.append({
            'gate':'MEAS',
            'ions': args
        })
    return outdata

def cleanup(indata):
    for inst in indata:
        if 'other' in inst.keys():
            if inst['other']==None:
                _ = inst.pop('other')
        if not (inst['gate']=='MEAS'):
            if not 'with_next' in inst.keys():
                inst['with_next'] = False

def ore_count_width(ore):
    count = 0
    for op in ore:
        if op['gate']=="ZZ":
            targ0 = op['ion']
            targ1 = op['other']
            targ = max(targ0, targ1)
            count = max(count, targ)
        elif op['gate'] in ['GPI', 'GPI2', 'GZ']:
            targ = op['ion']
            count = max(count, targ)
    
    return count+1


def ore_to_qiskit(ore, cwidth):
    i_measure = 0
    width = ore_count_width(ore)
    testqc = qkc.QuantumCircuit(width,cwidth)
    for op in ore:
        name = op['gate']
        if name=='GPI2':
            parm = op['phase']
            targ = op['ion']
            if parm==0.25:
                testqc.z(targ)
                testqc.h(targ)
            elif parm==0.75:
                testqc.h(targ)
                testqc.z(targ)
            elif parm==0.5:
                testqc.s(targ)
                testqc.h(targ)
                testqc.s(targ)
            elif parm==0.0:
                testqc.sdg(targ)
                testqc.h(targ)
                testqc.sdg(targ)
            else:
                raise ValueError("GPI2 param invalid")
        elif name=='ZZ':
            ctrl = op['ion']
            targ = op['other']
            if op['gain']!=1:
                raise ValueError("ZZ param invalid!")
            testqc.cz(ctrl, targ)
            testqc.s(ctrl)
            testqc.s(targ)
        
        elif name=='GZ':
            targ = op['ion']
            parm = op['phase']
            if parm==0.5:
                testqc.z(targ)
            elif parm==0.75:
                testqc.sdg(targ)
            elif parm==0.25:
                testqc.s(targ)
            elif parm==0.:
                pass
            else:
                raise ValueError("GZ param invalid!")
        
        elif name=='GPI':
            targ = op['ion']
            parm = op['phase']
            if parm==0.5:
                testqc.x(targ)
            elif parm==0.75:
                testqc.y(targ)
            elif parm==0.25:
                testqc.y(targ)
            elif parm==0.:
                testqc.x(targ)
            else:
                raise ValueError("GPI param invalid!")
        
        elif name=='MEAS':
            targs = op['ions']
            testqc.barrier()
            for q in targs:
                testqc.measure(q,i_measure)
                i_measure += 1
                testqc.reset(q)
            testqc.barrier()
        
        else:
            raise ValueError("Unrecognized gate")
    
    return testqc

def gen_echo(targs):
    ore = []
    for q in targs:
        op = {'gate': 'GPI', 'ion': q, 'phase': 0.5, 'with_next': False}
        ore.append(op)
    
    return ore

def gen_mcm(targs):
    ore = [
        {'gate': 'MEAS', 'ions': targs}
    ]
    return ore

def gen_h(targs, do_gz=True):
    ore = []
    carryphase = []
    for q in targs:
        op = {'gate': 'GPI2', 'ion': q, 'phase': 0.75, 'with_next': False}
        ore.append(op)
        carryphase.append(0.5)
    
    if not do_gz:
        return ore, carryphase
    else:
        for q,phase in zip(targs,carryphase):
            ore.append({'gate': 'GZ', 'ion': q, 'phase': phase, 'with_next': False})
        return ore

def syntax_xx(q0, q1):
    # !!!ASSUMES q0 is ancilla, q1 is data
    line0 = {"gate": "MS", "ion": q0, "other": q1, "phase": 0.0,\
        "with_next": True, 'filter': {'gateContext': 'cnL1R2'}}
    line1 = {"gate": "MS", "ion": q1, "other": q0, "phase": 0.0,\
        "with_next": False, 'filter': {'gateContext': 'cnR1L2'}}
    return line0, line1

def syntax_gpi2(q, phase, is_anc=False):
    ctx = {True: 'cnL1R2', False: 'cnR1L2'}[is_anc]
    return {'gate': 'GPI2', 'ion': q, 'phase': phase,\
        'with_next': False, 'filter': {'gateContext': ctx}}

def syntax_gpi(q, phase, is_anc=False):
    ctx = {True: 'cnL1R2', False: 'cnR1L2'}[is_anc]
    return {'gate': 'GPI' , 'ion': q, 'phase': phase,\
        'with_next': False, 'filter': {'gateContext': ctx}}

def syntax_gz(q, phase, is_anc=False):
    ctx = {True: 'cnL1R2', False: 'cnR1L2'}[is_anc]
    return {'gate': 'GZ'  , 'ion': q, 'phase': phase,\
        'with_next': False, 'filter': {'gateContext': ctx}}

def gen_z_checks(ancilla, qubits, do_gz=True):
    # Check that there are NO repeated targets
    qubitset = set(qubits)
    if len(qubitset) < len(qubits):
        raise ValueError("Bad qubit list")
    
    ore = []
    phase_anc = 0
    phase_dat = [0]*(len(qubits))

    ore.append(
        {'gate': 'GPI2', 'ion': ancilla, 'phase': 0.75, 'with_next': False}
    )
    for i,q in enumerate(qubits):
        zz_op = {'gate': 'ZZ',
            'ion': ancilla,
            'other': q,
            'phase': 0.0,
            'gain': 1,
            'with_next': False}
        ore.append(zz_op)
        phase_dat[i] += 3
    
    gpiphase = ((1-3*len(qubits))%4)*0.25
    ore.append(
        {'gate': 'GPI2', 'ion': ancilla, 'phase': gpiphase, 'with_next': False}
    )
    phase_anc += ((3*len(qubits))%4)*0.25
    phase_dat = [(phase%4)*0.25 for phase in phase_dat]

    if not do_gz:
        return ore, phase_anc, phase_dat
    else:
        ore.append({'gate': 'GZ', 'ion': ancilla, 'phase': phase_anc, 'with_next': False})
        for q,phase in zip(qubits,phase_dat):
            ore.append({'gate': 'GZ', 'ion': q, 'phase': phase, 'with_next': False})
        return ore

def gen_checks_XX(ancilla, qubits, basis='x'):
    # Check that there are NO repeated targets
    qubitset = set(qubits)
    if len(qubitset) < len(qubits):
        raise ValueError("Bad qubit list")

    _ = {'x':True, 'z':True}[basis] # Check valid basis
    
    ore = []
    ancilla_phase = 0

    if basis=='z': # Prepend basis rot. on data qb.
        for q in qubits:
            ore.append(syntax_gpi2(q, 0.25))

    for q in qubits: # Do XX gates.
        xx0, xx1 = syntax_xx(ancilla, q)
        ore.append(xx0)
        ore.append(xx1)
        ancilla_phase += 1
    
    # Final rotation on ancilla
    sandwich_phase = ancilla_phase % 4
    if sandwich_phase==1:
        ore.append(syntax_gpi2(ancilla, 0.5, True))
    elif sandwich_phase==2:
        ore.append(syntax_gpi(ancilla, 0., True))
    elif sandwich_phase==3:
        ore.append(syntax_gpi2(ancilla, 0., True))
    
    # Append basis rot. on data qb.
    if basis=='z':
        for q in qubits:
            ore.append(syntax_gpi2(q, 0.75))
            ore.append(syntax_gz(q, 0.75))
    elif basis=='x':
        for q in qubits:
            ore.append(syntax_gpi2(q, 0.5))
    return ore

def gen_x_checks(ancilla, qubits, do_gz=True):
    # Check that there are NO repeated targets
    qubitset = set(qubits)
    if len(qubitset) < len(qubits):
        raise ValueError("Bad qubit list")
    
    ore = []
    phase_anc = 0
    phase_dat = [0]*(len(qubits))

    ore.append(
        {'gate': 'GPI2', 'ion': ancilla, 'phase': 0.75, 'with_next': False}
    )
    for i,q in enumerate(qubits):
        gpi2_0_op = {'gate': 'GPI2', 'ion': q, 'phase': 0.75, 'with_next': False}
        gpi2_1_op = {'gate': 'GPI2', 'ion': q, 'phase': 0.50, 'with_next': False}
        zz_op = {'gate': 'ZZ',
            'ion': ancilla,
            'other': q,
            'phase': 0.0,
            'gain': 1,
            'with_next': False}
        ore.append(gpi2_0_op)
        ore.append(zz_op)
        ore.append(gpi2_1_op)
        phase_dat[i] += 3
    
    gpiphase = ((1-3*len(qubits))%4)*0.25
    ore.append(
        {'gate': 'GPI2', 'ion': ancilla, 'phase': gpiphase, 'with_next': False}
    )
    phase_anc += ((3*len(qubits))%4)*0.25
    phase_dat = [(phase%4)*0.25 for phase in phase_dat]

    if not do_gz:
        return ore, phase_anc, phase_dat
    else:
        ore.append({'gate': 'GZ', 'ion': ancilla, 'phase': phase_anc, 'with_next': False})
        for q,phase in zip(qubits,phase_dat):
            ore.append({'gate': 'GZ', 'ion': q, 'phase': phase, 'with_next': False})
        return ore

# def assemble_ore(oplist, argslist):

#     for op,arg in zip(oplist, argslist):
#         if op==''

# def write_to_file(fname, obj):
#     with open(fname, "w") as ofile:
#         json.dump(obj, ofile, indent=4)

def resolve_ore_phase(ore, width):
    '''Parse an ore gate list. Squash any lingering GZ'''
    out = []
    carryphase = [0]*width
    for op in ore:
        if op['gate']=='ZZ':
            out.append(op)
        elif op['gate']=='GZ':
            q = op['ion']
            p = op['phase']
            carryphase[q] = (carryphase[q]+p)%1
        elif op['gate']=='GPI':
            q = op['ion']
            carryphase[q] = {0.:0., 0.25:0.75, 0.75:0.25, 0.5:0.5}[carryphase[q]]
            out.append(op)
        elif op['gate']=='GPI2':
            q = op['ion']
            p = op['phase']
            newop = op
            newop['phase'] = (p-carryphase[q])%1
            out.append(newop)
        elif op['gate']=='MEAS':
            for q in op['ions']:
                carryphase[q] = 0
            out.append(op)
        else:
            raise ValueError("Unrecognized gate!")
    
    return out


# >>>>>> Buncha scripts to clean up Qiskit phases <<<<<< #
def glom_1q(op1, op2):
    if op1=='':
        return op2
    
    if op1=='z' and op2=='z':
        out = ''
    elif (op1=='z' and op2=='s') or (op2=='z' and op1=='s'):
        out = 'sdg'
    elif (op1=='z' and op2=='sdg') or (op2=='z' and op1=='sdg'):
        out = 's'
    elif (op1=='s' and op2=='s') or (op2=='sdg' and op1=='sdg'):
        out = 'z'
    return out

def append_1q(qc, lastoplist, qb):
    op = lastoplist[qb]
    lastoplist[qb] = ''
    if op=='z':
        qc.z(qb)
    elif op=='s':
        qc.s(qb)
    elif op=='sdg':
        qc.sdg(qb)

def clean_qiskit_phase(qc):
    qc0 = qc.copy_empty_like()
    qc0.dat = []
    lastop = ['']*len(qc.qubits)
    for op in qc:
        if op.name=='h':
            targ = qc.qubits.index(op.qubits[0])
            append_1q(qc0, lastop, targ)
            qc0.append(op)
        elif op.name=='cz':
            targs = [qc.qubits.index(q) for q in op.qubits]
            append_1q(qc0, lastop, targs[0])
            append_1q(qc0, lastop, targs[1])
            qc0.append(op)
        elif op.name=='measure':
            qc0.append(op)
        elif op.name in ['z','s','sdg']:
            targ = qc.qubits.index(op.qubits[0])
            glom_1q(lastop[targ], op.name)
        elif op.name in ['barrier', 'reset']:
            qc0.append(op)
        else:
            print(op.name)
            raise Exception("Bad gate")
    return qc0
# >>>>>> End: Buncha scripts to clean up Qiskit phases <<<<<< #

def count_stats(circuit):
    out = {}
    for gate in circuit:
        name = gate['gate']
        if name in out.keys():
            out[name] += 1
        else:
            out[name] = 1
    
    return out

def write_to_file(fname, compiled_ore):
    gates = []
    gates.append(dict(ore=compiled_ore))
    out_json = dict(gates=gates)

    with open(fname, "w") as ofile:
        ofile.write("module.exports = " + json.dumps(out_json, indent=2) + ";\n")

def read_from_file(fname):
    handle = open(fname, 'r')
    lines = handle.read()
    handle.close()

    json_string = lines.strip('module.exports = ').replace("};","}")
    ore = json.loads(json_string)

    return ore['gates'][0]['ore']