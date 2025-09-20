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


def updown_gen(max_dwin, tstart, fstart, cfg, reader, upchirp, downchirp, split, Nup, Ndown, FFT_N):
    retsup = xp.zeros((Nup + 1, FFT_N), dtype=xp.complex64)
    for i in range(Nup + 1):
        retsup[i] = dechirp_fft(tstart, fstart, reader, downchirp, i, True)

    retsdown = xp.zeros((Ndown + 1, FFT_N), dtype=xp.complex64)
    for i in range(Ndown + 1):
        retsdown[i] = dechirp_fft(tstart, fstart, reader, upchirp, i + cfg.sfdpos, False)

    pair_up = xp.abs(retsup[:-1, :-split]) + xp.abs(retsup[1:, split:])
    add_up = xp.sum(pair_up, axis=0)

    pair_down = xp.abs(retsdown[:-1, split:]) + xp.abs(retsdown[1:, :-split])
    add_down = xp.sum(pair_down, axis=0)

    yield 0, add_up, add_down # Yield the first result

    for dwin in range(1, max_dwin):
        # --- Update UP-CHIRP ---
        # Shift buffer: discard the oldest (index 0), move everything up.
        retsup[:-1] = retsup[1:]
        # Compute only the single NEW FFT required for this window.
        new_up_fft = dechirp_fft(tstart, fstart, reader, downchirp, Nup + dwin, True)
        # Add the new FFT to the end of the buffer.
        retsup[Nup] = new_up_fft
        # Recalculate sums efficiently from the updated buffer.
        pair_up = xp.abs(retsup[:-1, :-split]) + xp.abs(retsup[1:, split:])
        add_up = xp.sum(pair_up, axis=0)


        # --- Update DOWN-CHIRP ---
        # Shift buffer
        retsdown[:-1] = retsdown[1:]
        # Compute only the single NEW FFT.
        new_down_fft = dechirp_fft(tstart, fstart, reader, upchirp, Ndown + dwin + cfg.sfdpos, False)
        # Add the new FFT to the end of the buffer.
        retsdown[Ndown] = new_down_fft
        # Recalculate sums
        pair_down = xp.abs(retsdown[:-1, split:]) + xp.abs(retsdown[1:, :-split])
        add_down = xp.sum(pair_down, axis=0)

        yield dwin, add_up, add_down

def detect_slow(cfg, xp, fstart, tstart, dwin, updown_generator, bwnew2, FFT_N):
    lo    = int(xp.around((-cfg.bw - cfg.cfo_range) * FFT_N / cfg.fs)) + FFT_N // 2
    hi    = int(xp.around(cfg.cfo_range * FFT_N / cfg.fs)) + FFT_N // 2

    dwin_gen, add_up, add_down = next(updown_generator)
    assert dwin_gen == dwin

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
    est_to_s  = (t0 + dwin) * cfg.tsig + tstart 

    ret1 = float(to_scalar(xp.max(add_up[lo:hi])))
    ret2 = float(to_scalar(xp.max(add_down[lo:hi])))
    retval = ret1 + ret2
    return retval, float(est_cfo_f), float(est_to_s), ret1, ret2
