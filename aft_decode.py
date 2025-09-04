from functools import lru_cache
from utils import xp, around 

@lru_cache(maxsize=None)
def build_decode_matrices(n_classes: int, nsamp: int, fs: float, bw: float, sf: int):
    t = xp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    A = xp.zeros((n_classes, nsamp), dtype=xp.complex64)
    B = xp.zeros((n_classes, nsamp), dtype=xp.complex64)

    betai = bw / ((2 ** sf) / bw)
    wflag = True
    for code in range(n_classes):
        if (code - 1) % 4 != 0 and sf >= 11 and wflag:
            wflag = False
            continue
        nsamples = around(nsamp / n_classes * (n_classes - code))
        f01 = bw * (-0.5 + code / n_classes)
        ref1 = xp.exp(-1j * 2 * xp.pi * (f01 * t + 0.5 * betai * t * t))
        f02 = bw * (-1.5 + code / n_classes)
        ref2 = xp.exp(-1j * 2 * xp.pi * (f02 * t + 0.5 * betai * t * t))
        A[code, :nsamples] = ref1[:nsamples]
        if code > 0:
            B[code, nsamples:] = ref2[nsamples:]
    return A, B, t

def decode_payload(reader, Config, est_cfo_f: float, est_to_s: float):
    A, B, t = build_decode_matrices(Config.n_classes, Config.nsamp, Config.fs, Config.bw, Config.sf)

    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * (1 + 2 * est_cfo_f / Config.sig_freq)
    codes = []

    for pidx in range(Config.sfdend, Config.total_len):
        start_pos_all = (2 ** Config.sf / Config.bw) * Config.fs * (pidx + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = around(start_pos_all)
        dt = (start_pos - start_pos_all) / Config.fs

        dataX = reader.get(start_pos, Config.nsamp) * xp.exp(-1j * 2 * xp.pi * (est_cfo_f + betai * dt) * t)
        data1 = A @ dataX
        data2 = B @ dataX
        vals = xp.abs(data1) ** 2 + xp.abs(data2) ** 2
        coderet = int(xp.argmax(vals).item())
        codes.append(coderet)

    return codes
