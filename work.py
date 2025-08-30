from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from aft_decode import decode_payload






# freqs: before shift f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / n   if n is even
# after shift f = [-n/2, ..., -1, 0, 1, ...,   n/2-1] / n if n is even
# n is fft_n, f is cycles per sample spacing
# since fs=1e6: real freq in hz fhz=[-n/2, ..., -1, 0, 1, ...,   n/2-1] / n * 1e6Hz
# total range: sampling frequency. -fs/2 ~ fs/2, centered at 0
# bandwidth = 0.40625 sf
def myfft(chirp_data, n, plan):
    if USE_GPU:
        return xfft.fftshift(xfft.fft(chirp_data.astype(xp.complex64), n=n, plan=plan))
    else:
        return xfft.fftshift(xfft.fft(chirp_data.astype(xp.complex64), n=n))


def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = xp.exp(2j * xp.pi * est_cfo_freq * xp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    cfosymb = cfosymb.astype(xp.complex64)
    pktdata2a = pktdata_in * cfosymb
    return pktdata2a

def dechirp_fft(tstart, fstart, reader, refchirp, pidx, ispreamble):
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs# * (1 + fstart / Config.sig_freq)
    start_pos_all = nsamp_small * pidx + tstart
    start_pos = around(start_pos_all)
    start_pos_d = start_pos_all - start_pos
    sig1 = reader.get(start_pos, Config.nsamp)
    if sig1 is None or len(sig1) < len(refchirp):
        logger.warning(f"Too short {pidx=} reading from {start_pos} to {start_pos + Config.nsamp} {len(sig1)=} {len(refchirp)=}")
        return None
    sig2 = sig1 * refchirp
    freqdiff = start_pos_d / nsamp_small * Config.bw / Config.fs * Config.fft_n
    if ispreamble: freqdiff -= fstart / Config.sig_freq * Config.bw * pidx
    else: freqdiff += fstart / Config.sig_freq * Config.bw * pidx
    freqdiff -= fstart # TODO !!!
    sig2 = add_freq(sig2,freqdiff)
    data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
    return data0




def work(fstart, tstart, file_path):
    file_size = os.path.getsize(file_path)
    complex64_size = xp.dtype(xp.complex64).itemsize
    assert complex64_size == 8
    num_values = file_size // complex64_size // Config.nsamp
    print(f"{file_path=} Size in Number of symbols: {num_values}")

    reader = SlidingComplex64Reader(file_path, capacity=1_000_000, prefetch_ratio=0.25)
    x = xp.arange(Config.nsamp) * (1 + fstart / Config.sig_freq)
    bwnew = Config.bw * (1 + fstart / Config.sig_freq)
    bwnew2 = Config.bw * (1 - 2 * fstart / Config.sig_freq)
    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    betanew = beta * (1 + 2 * fstart / Config.sig_freq)
    upchirp = xp.exp(2j * xp.pi * (betanew / 2 * x ** 2 / Config.fs ** 2 + (- bwnew / 2) * x / Config.fs))
    downchirp = xp.conj(upchirp)
    # -------- constants ------------------------------------------------------
    Nup = Config.preamble_len  # - Config.skip_preambles  # 6
    FFT_N = Config.fft_n
    split = around(bwnew2 / Config.fs * FFT_N)  # index that
    lo = around((-Config.bw - Config.cfo_range) * FFT_N / Config.fs) + FFT_N // 2
    hi = around(Config.cfo_range * FFT_N / Config.fs) + FFT_N // 2

    up_fft_result = xp.empty((2, FFT_N), dtype=xp.complex64)
    pair_up = xp.empty((Nup, FFT_N - split), dtype=xp.float32)
    for i in range(Nup - 1):  # first six rows
        up_fft_result[i % 2, :] = dechirp_fft(tstart, fstart, reader, downchirp, i, True)
        if i > 0: pair_up[i - 1] = xp.abs(up_fft_result[(i - 1) % 2, :-split]) + xp.abs(up_fft_result[i % 2, split:])
    add_up = xp.sum(pair_up, axis=0)  # running total

    Ndown = 2
    down_fft_result = xp.empty((2, FFT_N), dtype=xp.complex64)
    pair_down = xp.empty((Ndown, FFT_N - split), dtype=xp.float32)
    for i in range(Ndown - 1):
        down_fft_result[i % 2, :] = dechirp_fft(tstart, fstart, reader, upchirp, i + Config.sfdpos, False)
        if i > 0: pair_down[i - 1] = xp.abs(down_fft_result[(i - 1) % 2, split:]) + xp.abs(
            down_fft_result[i % 2, :-split])
    add_down = xp.sum(pair_down, axis=0)  # running total

    # -------- detect-window sweep --------------------------------------------
    selected = []  # will store tuples (retval, detect_vals_array)
    WINDOW = 100
    K = 10
    detect_results_x = []
    detect_results_y = []  # for plotting all retval values
    res1 = []
    res2 = []
    for dwin in tqdm(range(400)):
        i = dwin + Nup - 1
        ret = dechirp_fft(tstart, fstart, reader, downchirp, i, True)
        if ret is None: break
        up_fft_result[i % 2, :] = ret
        add_up -= pair_up[i % Nup, :]
        pair_up[i % Nup, :] = xp.abs(up_fft_result[(i - 1) % 2, :-split]) + xp.abs(up_fft_result[i % 2, split:])
        add_up += pair_up[i % Nup, :]

        if False:
            nsamp_small = 2 ** Config.sf / Config.bw * Config.fs  # * (1 + fstart / Config.sig_freq)
            start_pos_all = nsamp_small * (i - Nup + 1) + tstart
            start_pos = around(start_pos_all)
            plt.plot(to_host(xp.unwrap(xp.angle(reader.get(start_pos, Config.nsamp * Nup)))))
            plt.title(f"Nup {dwin=}")
            plt.show()

        i = dwin + Ndown - 1
        ret = dechirp_fft(tstart, fstart, reader, upchirp, i + Config.sfdpos, False)
        if ret is None: break
        down_fft_result[i % 2, :] = ret
        add_down -= pair_down[i % Ndown, :]
        pair_down[i % Ndown, :] = xp.abs(down_fft_result[(i - 1) % 2, split:]) + xp.abs(down_fft_result[i % 2, :-split])
        add_down += pair_down[i % Ndown, :]

        if False:
            nsamp_small = 2 ** Config.sf / Config.bw * Config.fs  # * (1 + fstart / Config.sig_freq)
            start_pos_all = nsamp_small * (i - Ndown + 1 + Config.sfdpos) + tstart
            start_pos = around(start_pos_all)
            plt.plot(to_host(xp.unwrap(xp.angle(reader.get(start_pos, Config.nsamp * Ndown)))))
            plt.title(f"Nup {dwin=}")
            plt.show()

        # --- search peak in Σ|pair_up| (   Σ restricted to lo:hi   ) --------
        fup = to_host(xp.argmax(add_up[lo:hi])) + lo
        fu_sec = -1 if fup > -Config.bw // 2 * FFT_N / Config.fs + FFT_N // 2 else 1

        # --- best down-chirp over the two windows ----------------------------
        fdown = to_host(xp.argmax(add_down[lo:hi])) + lo
        # plt.plot(xp.abs(down_fft_result[0]).get())
        # plt.plot(xp.abs(down_fft_result[1]).get())
        # plt.axvline(x=fdown.item(), color='r', linestyle='dashed')
        # plt.show()
        #
        # plt.plot(xp.abs(add_up[lo:hi]).get())
        # plt.axvline(x=fup.item() - lo, color='r', linestyle='dashed')
        # plt.show()

        fd_sec = -1 if fdown > FFT_N // 2 else 1

        # print(fup, fu_sec, fdown, fd_sec, lo, hi)

        # --- CFO / TO grid (unchanged maths) ---------------------------------
        fft_up = (fup - FFT_N // 2) / (bwnew2 / Config.fs * FFT_N) + 0.5
        fft_down = (fdown - FFT_N // 2) / (bwnew2 / Config.fs * FFT_N) + 0.5
        fu = fft_up
        fd = fft_down
        f01 = (fu + fd) / 2
        f0 = (f01 + 0.5) % 1 - 0.5
        t0 = (f0 - fu - 0.5)
        t0 = t0 % 1 - 1
        est_cfo_f = f0 * Config.bw + fstart
        est_to_s = t0 * Config.tsig + tstart + dwin * (2 ** Config.sf / Config.bw * Config.fs)

        deltaf, deltat = xp.meshgrid(xp.array((0, fu_sec)), xp.array((0, fd_sec)))
        values = xp.zeros((2, 2, 3)).astype(float)
        nsamp_small = 2 ** Config.sf / Config.bw * Config.fs

        retval = xp.max(add_up[lo:hi]) + xp.max(add_down[lo:hi])
        res1.append(xp.max(add_up[lo:hi]))
        res2.append(xp.max(add_down[lo:hi]))
        detect_results_x.append(to_scalar(est_to_s))
        detect_results_y.append(to_scalar(retval))

        r, cfo, to = retval, est_cfo_f, est_to_s
        i = dwin

        # find conflicts within WINDOW
        conflicts = [p for p in selected if abs(p["i"] - i) < WINDOW]

        if not conflicts:
            if len(selected) < K:
                selected.append({"i": i, "r": r, "cfo": cfo, "to": to})
            else:
                worst = min(selected, key=lambda x: x["r"])
                if r > worst["r"]:
                    selected.remove(worst)
                    selected.append({"i": i, "r": r, "cfo": cfo, "to": to})
        else:
            # if better than the best in its neighborhood, replace it
            best_conf = max(conflicts, key=lambda x: x["r"])
            if r > best_conf["r"]:
                selected.remove(best_conf)
                selected.append({"i": i, "r": r, "cfo": cfo, "to": to})


    fig = pltfig1(detect_results_x, res1, title="plt res1")
    fig = pltfig1(detect_results_x, res2, title="plt res2", fig=fig)
    pltfig1(detect_results_x, detect_results_y, addvline=[x['to'] for x in selected], title="plt res all", fig=fig).show()

    anslist = []
    cfolist = []

    # report top-K by retval
    selected.sort(key=lambda x: x["r"], reverse=True)
    for p in selected:
        print(f"idx={p['i']:6d}  retval={p['r']:.6g}  est_cfo_f={p['cfo']:.6g}  est_to_s={p['to']:.6g}")
        retval = p['r']
        est_cfo_f = p['cfo']
        est_to_s = p['to']

        # est_cfo_f, est_to_s = refine_ft(est_cfo_f, est_to_s, reader)
        # logger.error(f"Rw f={est_cfo_f} t={est_to_s}")
        # est_cfo_f, est_to_s = find_power_new(est_cfo_f, est_to_s, reader)
        # logger.error(f"FF f={est_cfo_f} t={est_to_s}")
        plt.plot(to_host(xp.unwrap(xp.angle(reader.get(around(est_to_s) - Config.nsamp, Config.nsamp * 15)))))
        plt.axvline(Config.nsamp)
        plt.axvline(Config.nsamp * (Config.preamble_len + 1))
        plt.show()

        codes = decode_payload(reader, Config, est_cfo_f, est_to_s)
        logger.info(f"Result: f={est_cfo_f} t={est_to_s} Score={retval} Decoded results: {codes}")
