import analysis
from concurrent.futures import ProcessPoolExecutor
import dem_decode
decode_constructor = dem_decode.decoder_constructor

def _decode_key(args):
    key, circuit, dets, logs = args
    decoder = decode_constructor(circuit.detector_error_model())
    shots = len(dets)
    stat = []
    for i in range(shots):
        oneshot = analysis.decode_one(decoder, dets[i], logs[i])
        stat.append(oneshot)
    return key, stat


def decode_all(decode_dict, processes=1):
    work = [
        (key, decode_dict[key][3], decode_dict[key][1], decode_dict[key][2])
        for key in decode_dict.keys()
    ]

    if processes <= 1:
        return {key: stat for key, stat in map(_decode_key, work)}

    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = executor.map(_decode_key, work)
    return {key: stat for key, stat in results}
