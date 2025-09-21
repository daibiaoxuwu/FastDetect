import os
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from matplotlib import pyplot as plt
import math
from pre_detect import myfft
from symbtime import symbtime


def fitcoef2(coeff, coeft, reader):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi
    coeflist = []
    for pidx in range(0, Config.preamble_len):
        estf = xp.polyval(coeff, pidx)
        estbw = Config.bw * (1 + estf / Config.sig_freq)
        beta1 = betai * (1 + 2 * estf / Config.sig_freq)

        tstart = xp.polyval(coeft, pidx)
        tend = xp.polyval(coeft, pidx + 1)
        beta2 = 2 * xp.pi * (- estbw * 0.5 + estf) - tstart * 2 * beta1
        coef2d_est2 = sqlist([beta1, beta2, 0])
        nsymbr = xp.arange(math.ceil(tstart * Config.fs), math.ceil(tend * Config.fs))
        tsymbr = nsymbr / Config.fs
        sig = reader.get(to_scalar(nsymbr[0]), len(nsymbr))
        sig1 = sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        data0 = myfft(sig1, n=Config.fft_n, plan=Config.plan)
        freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
        freq, valnew = optimize_1dfreq_fast(sig1, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # logger.warning(f"{freq1=} {freq-freq1=} {valnew=}")
        coef2d_est2[1] = 2 * xp.pi * (- estbw * 0.5 + estf + freq) - tstart * 2 * beta1
        # sig2 = sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        # freq, valnew = optimize_1dfreq_Fast(sig2, tsymbr, freq1)
        # logger.warning(f"{freq=} should be zero {valnew=}")
        coef2d_est2[2] += xp.angle(sig.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))))
        coeflist.append(coef2d_est2)
    return xp.array(coeflist)


def fitcoef4(coeff: xp.array, coeft: xp.array, reader: SlidingComplex64Reader):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi
    coeflist = []
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs 

    for pidx in range(0, Config.preamble_len):
        estf = xp.polyval(coeff, pidx)
        estbw = Config.bw * (1 + estf / Config.sig_freq)
        beta1 = betai * (1 + 2 * estf / Config.sig_freq)

        tstart = xp.polyval(coeft, pidx)
        tend = xp.polyval(coeft, pidx + 1)
        beta2 = 2 * xp.pi * (- estbw * 0.5 + estf) - tstart * 2 * beta1
        coef2d_est2 = xp.array([to_scalar(beta1), to_scalar(beta2), 0])
        nsymbr_start = math.ceil(tstart * Config.fs + Config.nsamp / 8)
        nsymbr_end = math.ceil(tend * Config.fs - Config.nsamp / 8)
        nsymbr = xp.arange(math.ceil(tstart * Config.fs + Config.nsamp / 8), math.ceil(tend * Config.fs - Config.nsamp / 8))
        tsymbr = nsymbr / Config.fs
        sig0 = reader.get(nsymbr_start, nsymbr_end - nsymbr_start)
        sig1 = sig0 * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        data0 = myfft(sig1, n=Config.fft_n, plan=Config.plan)
        freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
        freq, valnew = optimize_1dfreq_fast(sig1, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # logger.warning(f"{freq1=} {freq-freq1=} {valnew=}")
        coef2d_est2[1] = 2 * xp.pi * (- estbw * 0.5 + estf + freq) - tstart * 2 * beta1
        # sig2 = sig * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        # freq, valnew = optimize_1dfreq_Fast(sig2, tsymbr, freq1)
        # logger.warning(f"{freq=} should be zero {valnew=}")
        coef2d_est2[2] += xp.angle(sig0.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))))
        coeflist.append(coef2d_est2)

    anslist = []
    for pidx in range(1, Config.preamble_len):
        tstart = xp.polyval(coeft, pidx)
        # print(pidx, coef2d_est2, wrap(xp.polyval(coef2d_est2, tstart)), wrap(xp.polyval(coef2d_est2, tend)))
        phasediff = wrap(xp.polyval(coeflist[pidx], tstart) - xp.polyval(coeflist[pidx - 1], tstart))
        anslist.append( to_scalar(phasediff ))
    anslist2 = xp.unwrap(xp.array(anslist))
    tdifflist = anslist2 / 2 / xp.pi / Config.bw
    # pltfig1(range(1, Config.preamble_len), anslist2).show()
    xrange = xp.arange(50, len(tdifflist))
    coefficients = xp.polyfit(xrange, tdifflist[xrange], 1)
    coeft_new = coeft.copy()
    coeft_new[-2:] += coefficients
    logger.warning(f"{coefficients=} {coeft=} {coeft_new=} cfo ppm from time: {1 - coeft_new[0] / Config.nsampf * Config.fs} cfo: {(1 - coeft_new[0] / Config.nsampf * Config.fs) * Config.sig_freq}")

    return coeft_new


def work_new(fstart, tstart, file_path):
    file_size = os.path.getsize(file_path)
    complex64_size = xp.dtype(xp.complex64).itemsize
    assert complex64_size == 8
    print(f"{file_path=} Size in Number of symbols: {file_size // complex64_size // Config.nsamp}")

    reader = SlidingComplex64Reader(file_path)
    nsamp_small_2 = 2 ** Config.sf / Config.bw * Config.fs * (1 - fstart / Config.sig_freq)

    # start_pos = around(tstart - 100)
    # length = 200
    # pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=(tstart,)).show()

    # tstart += nsamp_small_2 * 120
    # start_pos = around(tstart - 100)
    # print("Avg Amplitude:", xp.mean(xp.abs(reader.get(start_pos, length))))
    # pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=(tstart,)).show()

    # length = around(nsamp_small_2 + 200)
    # sig = reader.get(start_pos, length)
    # cfosymb = xp.exp(2j * xp.pi * fstart * xp.linspace(0, (len(sig) - 1) / Config.fs, num=len(sig)))
    # cfosymb = xp.unwrap(xp.angle(cfosymb)).astype(xp.float32)
    # sig = xp.unwrap(xp.angle(sig))
    # cfosymb += sig[100] - cfosymb[100]
    # fig = pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(sig), addvline=(tstart, tstart + nsamp_small_2))
    # pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(cfosymb), fig=fig).show()

    # length = 200
    # tstart += nsamp_small_2 * 120
    # start_pos = around(tstart - 100)
    # pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=(tstart,)).show()

    # tstart -= nsamp_small_2 * 240
    
    tsymblen = 2 ** Config.sf / Config.bw * (1 - fstart / Config.sig_freq)
    coeff = xp.array((0, fstart))
    coeft = xp.array((tsymblen, tstart / Config.fs))
    coeft = fitcoef4(coeff, coeft, reader)

    coeflist = fitcoef2(coeff, coeft, reader)
    coeff, coeft = symbtime(coeff, coeft, reader, coeflist)
    print(coeff, coeft)
