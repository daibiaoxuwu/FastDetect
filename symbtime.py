import math
import pickle
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from find_intersections import find_intersections
from decode_core import decode_core

def symbtime(coeff, coeft, reader, coeflist, margin=1000, nextstep=0):

    # see if the time estimations are correct
    # for pidx in xp.arange(10, Config.preamble_len, 20):
    #     tstart2 = xp.polyval(coeft, pidx)
    #     margin1 = 20
    #     nsymbr = xp.arange(around(tstart2 * Config.fs - margin1), around(tstart2 * Config.fs + margin1)).astype(xp.int64)
    #     tsymbr = nsymbr / Config.fs

    #     fig = pltfig1(tsymbr, xp.angle(reader.get(to_scalar(nsymbr[0]), len(nsymbr))), mode='markers', marker=dict(color='blue', size=4, symbol='circle'))
    #     fig = pltfig1(tsymbr, wrap(xp.polyval(coeflist[pidx - 1], tsymbr)), fig=fig)
    #     pltfig1(tsymbr, wrap(xp.polyval(coeflist[pidx], tsymbr)), title=f"pidx={pidx} phase diff", addvline=(tstart2,), fig=fig).show()

    # coarse estimation of range
    dx = []
    dy = []
    if False:
        for pidx in xp.arange(10, Config.preamble_len):
            tstart2 = xp.polyval(coeft, pidx)
            selected = find_intersections(coeflist[pidx - 1], coeflist[pidx], tstart2, reader, 1e-5, margin=margin, draw=False, remove_range=False) #!!! TODO remove range
            if selected != None:
                dx.append(pidx)
                dy.append(to_scalar(selected))
        dx = xp.array(dx)
        dy = xp.array(dy)
        with open(f"intersections0.pkl","wb") as f:
            pickle.dump((dx, dy), f)

    with open(f"intersections0.pkl","rb") as f:
        dx, dy = pickle.load(f)
    coeff_time = xp.polyfit(dx, dy, 1)

    logger.warning(f"guessed: {coeft=} coeff_time={coeff_time[0]:.12f},{coeff_time[1]:.12f} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
    pltfig(((dx, dy), (dx, xp.polyval(coeff_time, dx))), title="intersect points fitline").show()
    pltfig1(dx, dy - xp.polyval(coeff_time, dx), title="intersect points diff").show()

    dx2 = dy - xp.polyval(coeff_time, dx)
    dx2 = dx2[:225]
    dx = dx[:225]
    dx3 = dy.copy()[:225]
    for pidx in range(1, len(dx2) - 1):
        if abs(dx2[pidx] - dx2[pidx-1]) > 0.2e-6 and abs(dx2[pidx] + dx2[pidx-1]) > 0.2e-6:
            dx2[pidx] = (dx2[pidx-1] + dx2[pidx+1])/2
            dx3[pidx] = (dx3[pidx-1] + dx3[pidx+1])/2

    coeff_time2 = xp.polyfit(dx, dx2, 1)
    pltfig(((dx, dx2), (dx, xp.polyval(coeff_time2, dx))),
           title="intersect points fitline 2").show()
    pltfig1(dx, dx2 - xp.polyval(coeff_time2, dx), title="intersect points diff 2").show()
    logger.warning(f"coeff_time2={coeff_time2[0]:.12f},{coeff_time2[1]:.12f}")


    coeff_time3 = xp.polyfit(dx, dx3, 2)
    # pltfig(((dx, dx3), (dx, xp.polyval(coeff_time3, dx))),
    #        title="intersect points fitline coeff_time3").show()
    # pltfig1(dx, dx3 - xp.polyval(coeff_time3, dx), title="intersect points diff coeff_time3").show()
    logger.warning(f"coeff_time3={coeff_time3[0]:.18e},{coeff_time3[1]:.18e},{coeff_time3[2]:.18e}")
    # t1 - coef(1) - coef(0) = a + b
    freq_start = (1 - (coeff_time3[0] + coeff_time3[1]) / (2 ** Config.sf / Config.bw)) * Config.sig_freq
    # t2 - t1 = coef(2) - coef(1) - coef(1) + coef(0) = 2a
    freq_rate = - 2 * coeff_time3[0] / (2 ** Config.sf / Config.bw) * Config.sig_freq
    logger.warning(f"{freq_start=} {freq_rate=}")

    pidx_range = xp.arange(Config.preamble_len)

    # TODO simplify
    estcoefs = []
    for ixx in range(2):
        dd = []
        for pidx in range(240):
            estf = xp.polyval(coeff, pidx)
            if ixx == 0:
                bwdiff = -Config.bw * (1 + estf / Config.sig_freq) / 2
            else:
                bwdiff = Config.bw * (1 + estf / Config.sig_freq) / 2
            dd.append(to_scalar((coeflist[pidx, 0] * 2 * xp.polyval(coeff_time3, pidx + ixx) + coeflist[pidx, 1]) / 2 / xp.pi - bwdiff))
        dd = xp.array(dd)
        pidx_range2 = xp.arange(50, Config.preamble_len - 10)
        estcoef = xp.polyfit(pidx_range2, dd[pidx_range2], 1)
        intercept = xp.mean(dd[pidx_range2] - freq_rate * pidx_range2)
        estcoefs.append(estcoef)

        # pltfig(((pidx_range2, dd[pidx_range2]), (pidx_range2, xp.polyval(coeff_new, pidx_range2))),
        #        title=f"intersect points fitline freq{ixx}").show()
        # pltfig1(pidx_range2, dd[pidx_range2] - xp.polyval(coeff_new, pidx_range2), title=f"intersect points diff freq{ixx}").show()
        #
        fdiff = intercept - estcoef[1] # freq = (2at + b) / 2pi deltaf = a/pi deltat
        tdiff =  fdiff / xp.mean(coeflist[:, 0]) * xp.pi
        logger.warning(f"coef2 {'start' if ixx == 0 else 'end'} coeff_new at t=0: {estcoef[1]:.12f} estf change rate per symb: {estcoef[0]:.12f} fixed: {freq_rate:.12f} {intercept:.12f} {tdiff:.12f}")

    coeff_new, coeff_new1 = estcoefs

    coeff_time = coeft # todo!!!2
    # coeff_time[-1] += 0.4e-6 + 130e-9

    logger.warning(f"{xp.polyval(coeff_time, Config.preamble_len)=} {xp.polyval(coeff_time3, Config.preamble_len)=}")
    # coeff_time = coeff_time3 # !!! todo !!!
    # logger.warning(f"{coeff_time=} already replaced by coefftime3")

    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi
    coeffitlist = xp.zeros((Config.preamble_len, 3), dtype=xp.float64)
    coeffitlist[:, 0] = betai * (1 + 2 * xp.polyval(coeff_new, pidx_range) / Config.sig_freq)

    bwdiff = - Config.bw * (1 + coeff_new[1] / Config.sig_freq) / 2
    coeffitlist[:, 1] = 2 * xp.pi * xp.polyval(coeff_new, pidx_range) - xp.polyval(coeff_time, pidx_range) * 2 * coeffitlist[:, 0] + bwdiff * 2 * xp.pi

    for pidx in pidx_range[1:]:
        coeffitlist[pidx, 2] -= xp.polyval(coeffitlist[pidx], xp.polyval(coeff_time, pidx)) - xp.polyval(coeffitlist[pidx - 1], xp.polyval(coeff_time, pidx))


    codephase = []
    powers = []

    # preamble codephase and powers
    for pidx in range(Config.preamble_len):
        x1 = math.ceil(xp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(xp.polyval(coeff_time, pidx + 1) * Config.fs)
        nsymbr = xp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs
        sig = reader.get(x1, x2 - x1)
        res = sig.dot(xp.exp(-1j * xp.polyval(coeffitlist[pidx], tsymbr)))
        codephase.append(xp.angle(res).item())
        powers.append(xp.abs(res).item() / xp.sum(xp.abs(sig)).item())
    # pltfig1(None, xp.unwrap(codephase), title="unwrap phase").show()

    coeffitlist = xp.concatenate((coeffitlist, xp.zeros((100, 3))), axis=0)
    fig=None

    # pidx = -1
    # x1 = math.ceil(xp.polyval(coeff_time, pidx) * Config.fs)
    # x2 = math.ceil(xp.polyval(coeff_time, pidx + 2) * Config.fs)
    # print(x1, x2)
    # nsymbr = xp.arange(x1, x2)
    # sig = reader.get(x1, x2 - x1)
    # plt.plot(xp.unwrap(xp.angle(sig)).get())
    # plt.show()
    # sys.exit(0)

    for pidx in range(Config.preamble_len, Config.preamble_len + 2):

        # FFT find frequency
        x1 = math.ceil(xp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(xp.polyval(coeff_time, pidx + 1) * Config.fs)
        nsymbr = xp.arange(x1, x2)
        sig = reader.get(x1, x2 - x1)
        tsymbr = nsymbr / Config.fs

        estcoef_this = xp.polyval(coeff_new, pidx)
        beta1 = betai * (1 + 2 * estcoef_this / Config.sig_freq)
        estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
        beta2 = 2 * xp.pi * (estcoef_this - estbw / 2) - xp.polyval(coeff_time, pidx) * 2 * beta1 # 2ax+b=differential b=differential - 2 * beta1 * time
        coef2d_est = xp.array([beta1.get(), beta2.get(), 0])

        refchirp = xp.exp(-1j * xp.polyval(coef2d_est, tsymbr))
        sig2 = sig * refchirp
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        # plt.plot(xp.unwrap(xp.angle(sig)).get())
        # plt.title(f"preamble code uphase {pidx-Config.preamble_len}")
        # plt.show()
        freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
        freq, valnew = optimize_1dfreq(sig2, tsymbr, freq1, Config.fs / Config.fft_n * 5)  # valnew may be as low as 0.3, only half the power will be collected
        assert valnew > 0.9, f"FFT power <= 0.9, {pidx=} fft {freq=} {freq1=} FFTmaxpow={xp.max(xp.abs(data0))} {valnew=}"
        # plt.plot(xp.abs(data0).get())
        # plt.title(f"preamble code FFT {pidx-Config.preamble_len}")
        # plt.show()
        # freq, valnew = optimize_1dfreq(sig2, tsymbr, freq)
        code = freq / estbw * 2 ** Config.sf
        logger.warning(f"{freq1/ estbw * 2 ** Config.sf=} {freq/ estbw * 2 ** Config.sf=}")
        assert code >=0 and code < 4096
        # logger.warning(f"{pidx=} optimized fft {freq=} maxpow={valnew} {code=:.12f}")
        code = around(code)

        x3 = math.ceil(xp.polyval(coeff_time, pidx + 1 - code / 2 ** Config.sf ) * Config.fs)
        nsymbr1 = xp.arange(x1, x3)
        tsymbr1 = nsymbr1 / Config.fs
        sig21 = reader.get(x1, x3 - x1)

        beta2 = (2 * xp.pi * (xp.polyval(coeff_new, pidx) + estbw * (code / 2 ** Config.sf - 0.5))
                 - xp.polyval(coeff_time, pidx) * 2 * beta1)
        coef2d_est2 = xp.array([beta1.get(), beta2.get(), 0])
        coef2d_est2_2d = xp.polyval(coef2d_est2, xp.polyval(coeff_time, pidx)) - xp.polyval(coeffitlist[pidx - 1], xp.polyval(coeff_time, pidx))
        coef2d_est2[2] -= coef2d_est2_2d
        coeffitlist[pidx] = coef2d_est2

        res2 = sig21.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr1)) )

        codephase.append(xp.angle(res2).item())
        powers.append(xp.abs(res2).item() / xp.sum(xp.abs(sig21)).item())
        # pltfig1(tsymbr1, xp.angle(sig21 * xp.exp(-1j * 2 * xp.pi * freq1 * tsymbr1)), title=f"residue {pidx=}").show()
        logger.warning(f"{pidx=} {code=} {xp.angle(res2)=} pow={xp.abs(res2)/xp.sum(xp.abs(sig21))}")
        fig=pltfig1(tsymbr, xp.angle(sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}", fig=fig)

    for pidx in range(Config.preamble_len + 2, Config.preamble_len + 4):
        x1 = math.ceil(xp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(xp.polyval(coeff_time, pidx + 1) * Config.fs)
        sig = reader.get(x1, x2 - x1)
        nsymbr = xp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs

        estcoef_this = xp.polyval(coeff_new, pidx)
        beta1 = - betai * (1 + 2 * estcoef_this / Config.sig_freq)
        estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
        # logger.error(f"EEE! {Config.bw * (estcoef_this / Config.sig_freq)=}")
        beta2 = 2 * xp.pi * (xp.polyval(coeff_new, pidx) + estbw / 2) - xp.polyval(coeff_time, pidx) * 2 * beta1 # 2ax+b=differential b=differential - 2 * beta1 * time
        coef2d_est2 = xp.array([beta1.get(), beta2.get(), 0])
        coef2d_est2_2d = xp.polyval(coef2d_est2, xp.polyval(coeff_time, pidx)) - xp.polyval(
            coeffitlist[pidx - 1], xp.polyval(coeff_time, pidx))
        coef2d_est2[2] -= coef2d_est2_2d
        coeffitlist[pidx] = coef2d_est2
        res2 = sig.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr)))
        fig = pltfig1(tsymbr, xp.angle(sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}", fig=fig)
        codephase.append(xp.angle(res2).item())
        powers.append(xp.abs(res2).item() / xp.sum(xp.abs(sig)).item())
        # pltfig1(None, xp.angle(sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}").show()
        # freq, power = optimize_1dfreq(sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr)), tsymbr, 0)
        # logger.error(f"EEE! {freq=} {power=}")

    for pidx in range(Config.preamble_len + 4, Config.preamble_len + 5):
        x1 = math.ceil(xp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(xp.polyval(coeff_time, pidx + 0.25) * Config.fs)
        nsymbr = xp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs
        sig = reader.get(x1, x2 - x1)

        estcoef_this = xp.polyval(coeff_new, pidx)
        beta1 = - betai * (1 + 2 * estcoef_this / Config.sig_freq)
        estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
        beta2 = 2 * xp.pi * (xp.polyval(coeff_new, pidx) + estbw / 2) - xp.polyval(coeff_time, pidx) * 2 * beta1 # 2ax+b=differential b=differential - 2 * beta1 * time
        coef2d_est2 = xp.array([beta1.get(), beta2.get(), 0])
        coef2d_est2_2d = xp.polyval(coef2d_est2, xp.polyval(coeff_time, pidx)) - xp.polyval(
            coeffitlist[pidx - 1], xp.polyval(coeff_time, pidx))
        coef2d_est2[2] -= coef2d_est2_2d
        cd2 = xp.angle(sig.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr)) ))#!!!!!!!!!!!! TODO here we align phase for the last symbol
        logger.warning(f"WARN last phase {cd2=} manually add phase compensation")
        # coef2d_est2[2] += cd2
        # cd2 = xp.angle(sig.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr)) ))#!!!!!!!!!!!! TODO here we align phase for the last symbol
        # assert abs(cd2) < 1e-4
        coeffitlist[pidx] = coef2d_est2
        res2 = sig.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr)))
        codephase.append(xp.angle(res2).item())
        powers.append(xp.abs(res2).item() / xp.sum(xp.abs(sig)).item())
        fig=pltfig1(tsymbr, xp.angle(sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}", fig=fig)

    # coeff_time[1] -= 0.75 * coeff_time[0]
    # coeff_time[-1] -= 2.3e-6 #!!!!TODO!!!!!a
    # logger.warning(f"{xp.polyval(coeff_time, Config.preamble_len + 5)=:.12e}")
    # logger.warning(f"{xp.polyval(coeff_time3, Config.preamble_len + 5 - 0.75)=:.12e}")

    startphase = xp.polyval(coeffitlist[Config.preamble_len + 4], xp.polyval(coeff_time, Config.preamble_len + 5 - 0.75))

    if nextstep:
        coef2d_ests = []
        ifreqs = []
        # for pidx in range(Config.preamble_len + 5, Config.preamble_len + 5 + Config.payload_len):
        codephase2 = [] # !!! todo !!!
        powers = []
        codes = []
        for pidx in range(Config.preamble_len + 5, Config.total_len):
            tstart = xp.polyval(coeff_time, pidx - 0.75)
            tend = xp.polyval(coeff_time, pidx + 1 - 0.75)
            x1 = math.ceil(tstart * Config.fs)
            x2 = math.ceil(tend * Config.fs)
            nsymbr = xp.arange(x1, x2)
            sig = reader.get(x1, x2 - x1)
            if xp.mean(xp.abs(sig)) < 0.01:
                logger.error(f"{pidx=} {xp.mean(xp.abs(sig))=} too small. is symbol ending? quitting, payload_len={pidx - Config.preamble_len - 5}")
                break
            code, endphase, coef2d_est2, coef2d_est2a, res2, res2a, ifreq1, ifreq2 = decode_core(reader, tstart, tend, coeff, startphase, pidx)
            startphase = endphase
            powers.append(xp.abs(res2).item())
            powers.append(xp.abs(res2a).item())
            codephase2.append(xp.angle(res2).item())
            codephase2.append(xp.angle(res2a).item())
            codephase.append(xp.angle(res2).item())
            codephase.append(xp.angle(res2a).item())
            coef2d_ests.append(coef2d_est2)
            coef2d_ests.append(coef2d_est2a)
            codes.append(code)

        anslist = []
        anslista = []
        anslistb = []
        anslist2 = []
        anslist2a = []
        anslist2b = []
        for pidx in range(2, len(codephase2), 2):
            code = codes[pidx // 2]
            tmid = tstart * (code / 2 ** Config.sf) + tend * (1 - code / 2 ** Config.sf)
            tmid = tmid.item()
            ifreq1 = xp.polyval(sqlist([2 * coef2d_ests[pidx][0], coef2d_ests[pidx][1]]), tstart ) - xp.polyval(sqlist([2 * coef2d_ests[pidx - 1][0], coef2d_ests[pidx - 1][1]]), tstart )
            ifreq2 = xp.polyval(sqlist([2 * coef2d_ests[pidx + 1][0], coef2d_ests[pidx + 1][1]]), tmid ) - xp.polyval(sqlist([2 * coef2d_ests[pidx][0], coef2d_ests[pidx][1]]), tmid )
            print(pidx, ifreq1, ifreq2)
            a1 = (wrap(codephase2[pidx] - codephase2[pidx - 1] - xp.pi) + xp.pi) / 2 / xp.pi / ifreq1
            if ifreq1 < 0: a1 = (wrap(codephase2[pidx] - codephase2[pidx - 1] + xp.pi) - xp.pi) / 2 / xp.pi / ifreq1
            assert a1>=0
            a1a = a1 + 1 / abs(ifreq1)
            a1b = a1 - 1 / abs(ifreq1)
            anslist.append(a1)
            anslista.append(a1a)
            anslistb.append(a1b)
            a2 = wrap(codephase2[pidx + 1] - codephase2[pidx]) / 2 / xp.pi / ifreq2
            a2a = a2 + 1 / abs(ifreq2)
            a2b = a2 - 1 / abs(ifreq2)
            anslist2.append(a2)
            anslist2a.append(a2a)
            anslist2b.append(a2b)

        anslist = xp.unwrap(sqlist(anslist))
        tdifflist = anslist
        fig = pltfig1(None, tdifflist, title="tdifflist")
        fig = pltfig1(None, anslista, title="tdifflist", fig=fig)
        pltfig1(None, anslistb, title="tdifflista", fig=fig).show()
        anslist2 = xp.unwrap(sqlist(anslist2))
        tdifflist2 = anslist2
        fig = pltfig1(None, tdifflist2, title="tdifflist")
        fig = pltfig1(None, anslist2a, title="tdifflist", fig=fig)
        pltfig1(None, anslist2b, title="tdifflista", fig=fig).show()


        pltfig1(None, powers, title="powers").show()
        pltfig1(None, xp.unwrap(codephase), title="unwrap phase").show()
    return coeff_new, coeff_time
