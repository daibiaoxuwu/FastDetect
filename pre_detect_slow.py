from utils import xp, xfft, around, USE_GPU, to_host, to_scalar
from Config import Config
from matplotlib import pyplot as plt

# ---------- helpers already in your file ----------

def myfft(chirp_data, n, plan):
    if USE_GPU:
        return xfft.fftshift(xfft.fft(chirp_data.astype(xp.complex64), n=n, plan=plan))
    else:
        return xfft.fftshift(xfft.fft(chirp_data.astype(xp.complex64), n=n))

def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = xp.exp(2j * xp.pi * est_cfo_freq * xp.linspace(
        0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    cfosymb = cfosymb.astype(xp.complex64)
    return pktdata_in * cfosymb

def dechirp_fft(tstart, fstart, reader, refchirp, pidx, ispreamble):
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    start_pos_all = nsamp_small * pidx + tstart
    start_pos = around(start_pos_all)
    start_pos_d = start_pos_all - start_pos

    sig1 = reader.get(start_pos, Config.nsamp)
    if sig1 is None or len(sig1) < len(refchirp):
        return None

    sig2 = sig1 * refchirp
    freqdiff = start_pos_d / nsamp_small * Config.bw / Config.fs * Config.fft_n
    if ispreamble:
        freqdiff -= fstart / Config.sig_freq * Config.bw * pidx
    else:
        freqdiff += fstart / Config.sig_freq * Config.bw * pidx
    freqdiff -= fstart  # TODO ???

    sig2 = add_freq(sig2, freqdiff)
    return myfft(sig2, n=Config.fft_n, plan=Config.plan)

def detect_slow_init(cfg, xp, fstart, tstart, reader):
    # ---- precompute chirps/consts (once) ----
    x = xp.arange(cfg.nsamp) * (1 + fstart / cfg.sig_freq)
    bwnew  = cfg.bw * (1 + fstart / cfg.sig_freq)
    bwnew2 = cfg.bw * (1 - 2 * fstart / cfg.sig_freq)
    beta        = cfg.bw / ((2 ** cfg.sf) / cfg.bw)
    betanew     = beta * (1 + 2 * fstart / cfg.sig_freq)

    upchirp   = xp.exp(2j * xp.pi * (betanew / 2 * x ** 2 / cfg.fs ** 2 + (-bwnew / 2) * x / cfg.fs))
    downchirp = xp.conj(upchirp)

    Nup   = cfg.preamble_len
    Ndown = 2
    FFT_N = cfg.fft_n
    split = int(xp.around(bwnew2 / cfg.fs * FFT_N))
    lo    = int(xp.around((-cfg.bw - cfg.cfo_range) * FFT_N / cfg.fs)) + FFT_N // 2
    hi    = int(xp.around(cfg.cfo_range * FFT_N / cfg.fs)) + FFT_N // 2

    # ---- allocate sliding buffers (filled in warm_start) ----
    pair_up   = xp.zeros((Nup,   FFT_N - split), dtype=xp.float32)
    pair_down = xp.zeros((Ndown, FFT_N - split), dtype=xp.float32)

    # Fill UP: compute Nup-1 FFTs, build pair_up rows [0..Nup-2)
    retsup = xp.zeros((Nup + 1, FFT_N), dtype=xp.complex64)
    retsdown = xp.zeros((Nup + 1, FFT_N), dtype=xp.complex64)
    for i in range(Nup + 1):
        retsup[i] = dechirp_fft(tstart, fstart, reader, downchirp, i, True)
    for i in range(Nup):
        pair_up[i] = xp.abs(retsup[i, : -split]) + xp.abs(retsup[i + 1, split :]) # TODO need smaller slice?

    add_up = xp.sum(pair_up, axis=0)

    # Fill DOWN: compute Ndown-1 FFTs (i.e. 1), build pair_down row 0
    for i in range(Ndown + 1):
        retsdown[i] = dechirp_fft(tstart, fstart, reader, upchirp, i + cfg.sfdpos, False)
    for i in range(Ndown):
        pair_down[i] = xp.abs(retsdown[i, split :]) + xp.abs(retsdown[i + 1, : -split])


    add_down = xp.sum(pair_down, axis=0)

    # --- peak search in restricted band ---
    fup   = int(to_scalar(xp.argmax(add_up[lo:hi]))) + lo
    fdown = int(to_scalar(xp.argmax(add_down[lo:hi]))) + lo

    fft_up   = (fup   - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fft_down = (fdown - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fu = fft_up + 0.5
    fd = fft_down - 0.5

    f01 = (fu + fd) / 2
    f0  = (f01 + 0.5) % 1 - 0.5
    t0  = (f0 - fu)
    t0  = t0 % 1

    est_cfo_f = f0 * cfg.bw + fstart
    est_to_s  = t0 * cfg.tsig + tstart 

    ret1 = float(to_scalar(xp.max(add_up[lo:hi])))
    ret2 = float(to_scalar(xp.max(add_down[lo:hi])))
    retval = ret1 + ret2
    return retval, float(est_cfo_f), float(est_to_s), ret1, ret2, retsup, retsdown

def detect_slow(cfg, xp, fstart, tstart, reader, retsup_in, retsdown_in):
    # ---- precompute chirps/consts (once) ----
    x = xp.arange(cfg.nsamp) * (1 + fstart / cfg.sig_freq)
    bwnew  = cfg.bw * (1 + fstart / cfg.sig_freq)
    bwnew2 = cfg.bw * (1 - 2 * fstart / cfg.sig_freq)
    beta        = cfg.bw / ((2 ** cfg.sf) / cfg.bw)
    betanew     = beta * (1 + 2 * fstart / cfg.sig_freq)

    upchirp   = xp.exp(2j * xp.pi * (betanew / 2 * x ** 2 / cfg.fs ** 2 + (-bwnew / 2) * x / cfg.fs))
    downchirp = xp.conj(upchirp)

    Nup   = cfg.preamble_len
    Ndown = 2
    FFT_N = cfg.fft_n
    split = int(xp.around(bwnew2 / cfg.fs * FFT_N))
    lo    = int(xp.around((-cfg.bw - cfg.cfo_range) * FFT_N / cfg.fs)) + FFT_N // 2
    hi    = int(xp.around(cfg.cfo_range * FFT_N / cfg.fs)) + FFT_N // 2

    # ---- allocate sliding buffers (filled in warm_start) ----
    pair_up   = xp.zeros((Nup,   FFT_N - split), dtype=xp.float32)
    pair_down = xp.zeros((Ndown, FFT_N - split), dtype=xp.float32)

    # Fill UP: compute Nup-1 FFTs, build pair_up rows [0..Nup-2)
    retsup = xp.roll(retsup_in, shift=-1, axis=0)
    retsup[Nup] = dechirp_fft(tstart, fstart, reader, downchirp, Nup, True)
    for i in range(Nup):
        pair_up[i] = xp.abs(retsup[i, : -split]) + xp.abs(retsup[i + 1, split :]) # TODO need smaller slice?

    add_up = xp.sum(pair_up, axis=0)

    retsdown = xp.roll(retsdown_in, shift=-1, axis=0)
    retsdown[Ndown] = dechirp_fft(tstart, fstart, reader, upchirp, Ndown + cfg.sfdpos, False)
    for i in range(Ndown):
        pair_down[i] = xp.abs(retsdown[i, split :]) + xp.abs(retsdown[i + 1, : -split])


    add_down = xp.sum(pair_down, axis=0)

    # --- peak search in restricted band ---
    fup   = int(to_scalar(xp.argmax(add_up[lo:hi]))) + lo
    fdown = int(to_scalar(xp.argmax(add_down[lo:hi]))) + lo

    fft_up   = (fup   - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fft_down = (fdown - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fu = fft_up + 0.5
    fd = fft_down - 0.5

    f01 = (fu + fd) / 2
    f0  = (f01 + 0.5) % 1 - 0.5
    t0  = (f0 - fu)
    t0  = t0 % 1

    est_cfo_f = f0 * cfg.bw + fstart
    est_to_s  = t0 * cfg.tsig + tstart 

    ret1 = float(to_scalar(xp.max(add_up[lo:hi])))
    ret2 = float(to_scalar(xp.max(add_down[lo:hi])))
    retval = ret1 + ret2
    return retval, float(est_cfo_f), float(est_to_s), ret1, ret2