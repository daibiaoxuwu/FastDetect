from utils import xp, xfft, around, USE_GPU, to_host, to_scalar
from Config import Config

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


def detect_slow(cfg, xp, fstart, tstart, reader):
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
    up_fft_result   = xp.empty((2, FFT_N), dtype=xp.complex64)
    down_fft_result = xp.empty((2, FFT_N), dtype=xp.complex64)
    pair_up   = xp.zeros((Nup,   FFT_N - split), dtype=xp.float32)
    pair_down = xp.zeros((Ndown, FFT_N - split), dtype=xp.float32)
    add_down  = None

    # Fill UP: compute Nup-1 FFTs, build pair_up rows [0..Nup-2)
    for i in range(Nup):
        ret1 = dechirp_fft(tstart, fstart, reader, downchirp, i, True)
        ret2 = dechirp_fft(tstart, fstart, reader, downchirp, i + 1, True)
        pair_up[i] = xp.abs(ret1[ : -split]) + xp.abs(ret2[split :])

    add_up = xp.sum(pair_up, axis=0)

    # Fill DOWN: compute Ndown-1 FFTs (i.e. 1), build pair_down row 0
    for i in range(Ndown):
        ret1 = dechirp_fft(tstart, fstart, reader, upchirp, i + cfg.sfdpos, False)
        ret2 = dechirp_fft(tstart, fstart, reader, upchirp, i + cfg.sfdpos + 1, False)
        pair_down[i] = xp.abs(ret1[split :]) + xp.abs(ret2[: -split])

    add_down = xp.sum(pair_down, axis=0)

    # --- peak search in restricted band ---
    fup   = int(to_scalar(xp.argmax(add_up[lo:hi]))) + lo
    fu_sec = -1 if fup > -cfg.bw // 2 * FFT_N / cfg.fs + FFT_N // 2 else 1

    fdown = int(to_scalar(xp.argmax(add_down[lo:hi]))) + lo
    fd_sec = -1 if fdown > FFT_N // 2 else 1

    fft_up   = (fup   - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fft_down = (fdown - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fu = fft_up
    fd = fft_down

    f01 = (fu + fd) / 2
    f0  = (f01 + 0.5) % 1 - 0.5
    t0  = (f0 - fu - 0.5)
    t0  = t0 % 1 - 1

    est_cfo_f = f0 * cfg.bw + fstart
    est_to_s  = t0 * cfg.tsig + tstart 

    ret1 = float(to_scalar(xp.max(add_up[lo:hi])))
    ret2 = float(to_scalar(xp.max(add_down[lo:hi])))
    retval = ret1 + ret2
    return retval, float(est_cfo_f), float(est_to_s), ret1, ret2

