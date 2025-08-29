from utils import xp, xfft, USE_GPU, around
import argparse

parser = argparse.ArgumentParser(description="Sample argparse script")
parser.add_argument('--sf', type=int, default=10, help="Set the value of sf (default is 7)")
args = parser.parse_args()

class Config:
    sf = args.sf
    bw = 125000
    sig_freq = 903.9e6
    preamble_len = 8
    total_len = 30
    guess_f = 0
    fs = 1e6
    skip_preambles = 2  # skip first 8 preambles ## TODO
    code_len = 2

    cfo_range = bw // 4
    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = around(n_classes * fs / bw)
    nsampf = (n_classes * fs / bw)

    tstandard = xp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    decode_matrix_a = xp.zeros((n_classes, nsamp), dtype=xp.complex64)
    decode_matrix_b = xp.zeros((n_classes, nsamp), dtype=xp.complex64)

    betai = bw / ((2 ** sf) / bw)
    wflag = True
    for code in range(n_classes):
        if (code-1)%4!=0 and sf>=11 and wflag:
            wflag = False
            continue
        nsamples = around(nsamp / n_classes * (n_classes - code))
        f01 = bw * (-0.5 + code / n_classes)
        refchirpc1 = xp.exp(-1j * 2 * xp.pi * (f01 * tstandard + 0.5 * betai * tstandard * tstandard))
        f02 = bw * (-1.5 + code / n_classes)
        refchirpc2 = xp.exp(-1j * 2 * xp.pi * (f02 * tstandard + 0.5 * betai * tstandard * tstandard))
        decode_matrix_a[code, :nsamples] = refchirpc1[:nsamples]
        if code > 0: decode_matrix_b[code, nsamples:] = refchirpc2[nsamples:]

    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3

    detect_range_pkts = 1000 # !!! TODO
    fft_n = int(fs)
    if USE_GPU:
        plan = xfft.get_fft_plan(xp.zeros(fft_n, dtype=xp.complex64))
        plan2 = xfft.get_fft_plan(xp.zeros(nsamp, dtype=xp.complex64))
    else:
        plan = None
        plan2 = None