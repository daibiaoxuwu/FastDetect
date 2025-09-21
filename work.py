from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from aft_decode import decode_payload
# from pre_detect import FastDetectContext
from pre_detect import detect_slow, updown_gen

def mychirp(t, f0, f1, t1):
    beta = (f1 - f0) / t1
    phase = 2 * xp.pi * (f0 * t + 0.5 * beta * t * t)
    sig = xp.exp(1j * to_device(phase))
    return sig

def work(fstart, tstart, file_path):
    file_size = os.path.getsize(file_path)
    complex64_size = xp.dtype(xp.complex64).itemsize
    assert complex64_size == 8
    print(f"{file_path=} Size in Number of symbols: {file_size // complex64_size // Config.nsamp}")

    reader = SlidingComplex64Reader(file_path)
   

    # -------- detect-window sweep --------------------------------------------
    selected = []  # will store tuples (retval, detect_vals_array)
    WINDOW = 100
    K = 10
    detect_results_x = []
    detect_results_y = []  # for plotting all retval values
    res1 = []
    res2 = []
    # fast_detect = FastDetectContext(Config, xp, fstart)
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    max_dwin = (file_size // complex64_size - around(nsamp_small * (2 + Config.sfdpos + 1) + tstart) + Config.nsamp) // Config.nsamp
    # max_dwin = min(500, max_dwin)  # !!!
    x = xp.arange(Config.nsamp) * (1 + fstart / Config.sig_freq)
    bwnew2 = Config.bw * (1 - 2 * fstart / Config.sig_freq)
    beta        = Config.bw / ((2 ** Config.sf) / Config.bw)
    betanew     = beta * (1 + 2 * fstart / Config.sig_freq)
    bwnew = Config.bw * (1 + fstart / Config.sig_freq)

    upchirp = xp.exp(2j * xp.pi * (betanew / 2 * x ** 2 / Config.fs ** 2 + (- bwnew / 2) * x / Config.fs))
    downchirp = xp.conj(upchirp)

    Nup   = Config.preamble_len
    Ndown = 2
    FFT_N = Config.fft_n
    split = int(xp.around(bwnew2 / Config.fs * FFT_N))

    start_dwin = 396
    updown_generator = updown_gen(max_dwin, tstart, fstart, Config, reader, upchirp, downchirp, split, Nup, Ndown, FFT_N, start_dwin)

    list1 = []
    list2 = []
    list3 = []
    for dwin in tqdm(range(start_dwin,694)):  # !!! TODO max_dwin
        r, cfo, to, res1x, res2x, temp_result = detect_slow(Config, xp, fstart, tstart, dwin, updown_generator, bwnew2, FFT_N)
        list1v, list2v, list3v = temp_result

        if r is None: break
        res1.append(res1x)
        res2.append(res2x)
        detect_results_x.append(dwin)#to_scalar(to)) # !!! TODO
        detect_results_y.append(to_scalar(r))
        list1.append(to_scalar(list1v))
        list2.append(to_scalar(list2v[0]))
        list3.append(to_scalar(list3v))

        # find conflicts within WINDOW
        conflicts = [p for p in selected if abs(p["i"] - dwin) < WINDOW]

        if not conflicts:
            if len(selected) < K:
                selected.append({"i": dwin, "r": r, "cfo": cfo, "to": to})
            else:
                worst = min(selected, key=lambda x: x["r"])
                if r > worst["r"]:
                    selected.remove(worst)
                    selected.append({"i": dwin, "r": r, "cfo": cfo, "to": to})
        else:
            # if better than the best in its neighborhood, replace it
            best_conf = max(conflicts, key=lambda x: x["r"])
            if r > best_conf["r"]:
                selected.remove(best_conf)
                selected.append({"i": dwin, "r": r, "cfo": cfo, "to": to})


    fig = pltfig1(detect_results_x, res1, title="plt res1")
    fig = pltfig1(detect_results_x, res2, title="plt res2", fig=fig)
    # pltfig1(detect_results_x, detect_results_y, addvline=[x['to'] for x in selected], title="plt res all", fig=fig).show()
    pltfig1(detect_results_x, detect_results_y, title="plt res all", fig=fig).show()

    fig = pltfig1(detect_results_x, list1)
    pltfig1(detect_results_x, list3, title="plt abs_retsup1_at_argmax_pair_up", fig=fig).show()
    pltfig1(detect_results_x, list2, title="plt argmax_pair_up").show()

    # report top-K by retval
    selected.sort(key=lambda x: x["r"], reverse=True)
    # selected.sort(key=lambda x: x["to"], reverse=False)
    for p in selected[:1]: # !!! TODO only the fixed one
        print(f"idx={p['i']:6d}  retval={p['r']:.6g}  est_cfo_f={p['cfo']:.6g}  est_to_s={p['to']:.6g}")
        retval = p['r']
        est_cfo_f = fstart # p['cfo']
        bwnew2 = Config.bw * (1 - 2 * est_cfo_f / Config.sig_freq)
        nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
        nsamp_small_2 = 2 ** Config.sf / Config.bw * Config.fs * (1 - est_cfo_f / Config.sig_freq)
        est_to_s = 4240090.873306715 # p['to'] 
        print(nsamp_small, nsamp_small_2, nsamp_small - nsamp_small_2)

        # est_cfo_f, est_to_s = refine_ft(est_cfo_f, est_to_s, reader)
        # logger.error(f"Rw f={est_cfo_f} t={est_to_s}")
        # est_cfo_f, est_to_s = find_power_new(est_cfo_f, est_to_s, reader)
        # logger.error(f"FF f={est_cfo_f} t={est_to_s}")

        start_pos = around(est_to_s - nsamp_small_2)
        length = Config.nsamp * 7
        print("Avg Amplitude:", xp.mean(xp.abs(reader.get(start_pos, length))))
        pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=to_host(xp.arange(around(est_to_s), around(est_to_s) + nsamp_small_2 * 6, nsamp_small_2))).show()

        est_to_s += nsamp_small_2 * 120
        start_pos = around(est_to_s - nsamp_small_2)
        print("fin", est_to_s + nsamp_small_2)
        pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=to_host(xp.arange(around(est_to_s), around(est_to_s) + nsamp_small_2 * 6, nsamp_small_2))).show()

        est_to_s += nsamp_small_2 * 120
        start_pos = around(est_to_s - nsamp_small_2)
        pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=to_host(xp.arange(around(est_to_s), around(est_to_s) + nsamp_small_2 * 6, nsamp_small_2))).show()
        # plt.axvline(nsamp_small_2 * (Config.preamble_len + 1))
        # plt.axvline(nsamp_small_2 * (Config.preamble_len + 1 + 2.25))
        # plt.axvline(nsamp_small_2 * (Config.total_len + 1))

        # start_pos = around(est_to_s)
        # length = Config.nsamp * 50
                            
        # plt.plot(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length))))) # !!! TODO 15
        # plt.axvline(around(est_to_s) + nsamp_small_2)
        # plt.axvline(around(est_to_s) + nsamp_small_2 * (Config.preamble_len + 2))
        # plt.axvline(around(est_to_s) + nsamp_small_2 * (Config.preamble_len + 4.25))
        # plt.show()

        logger.info(f"Result: f={est_cfo_f} t={est_to_s} Score={retval}")
        # codes = decode_payload(reader, Config, est_cfo_f, est_to_s, False)
        # logger.info(f"Linear Decoded results: {codes}")
        # codes = decode_payload(reader, Config, est_cfo_f, est_to_s, True)
        # logger.info(f"Curvin Decoded results: {codes}")
