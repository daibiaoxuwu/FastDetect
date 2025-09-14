from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from aft_decode import decode_payload
# from pre_detect import FastDetectContext
from pre_detect_slow import detect_slow


def work(fstart, tstart, file_path):
    file_size = os.path.getsize(file_path)
    complex64_size = xp.dtype(xp.complex64).itemsize
    assert complex64_size == 8
    num_values = file_size // complex64_size // Config.nsamp
    print(f"{file_path=} Size in Number of symbols: {num_values}")

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

    for dwin in range(2000): # !!!
        r, cfo, to, res1x, res2x = detect_slow(Config, xp, fstart, tstart + dwin * Config.nsamp, reader)
        if r is None: break
        res1.append(res1x)
        res2.append(res2x)
        detect_results_x.append(to_scalar(to))
        detect_results_y.append(to_scalar(r))

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
    pltfig1(detect_results_x, detect_results_y, addvline=[x['to'] for x in selected], title="plt res all", fig=fig).show()

    # report top-K by retval
    # selected.sort(key=lambda x: x["r"], reverse=True)
    selected.sort(key=lambda x: x["to"], reverse=False)
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
        plt.axvline(Config.nsamp * (Config.preamble_len + 1 + 2.25))
        plt.show()

        logger.info(f"Result: f={est_cfo_f} t={est_to_s} Score={retval}")
        codes = decode_payload(reader, Config, est_cfo_f, est_to_s, False)
        logger.info(f"Linear Decoded results: {codes}")
        codes = decode_payload(reader, Config, est_cfo_f, est_to_s, True)
        logger.info(f"Curvin Decoded results: {codes}")
