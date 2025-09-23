import os
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from matplotlib import pyplot as plt
import math
from symbtime import symbtime


# compute all coef2d for each preamble symbol, for symbtime use. 
def fitcoef2(coeff: xp.array, coeft: xp.array, reader: SlidingComplex64Reader):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi # frequency slope to phase 2d slope, *pi
    coeflist = []

    for pidx in range(0, Config.preamble_len):
        
        # compute coef2d_est2: polynomial curve fitting unwrapped phase of symbol pidx
        # time: tstart to tend
        # frequency at tstart: - estbw * 0.5 + estf
        estf = xp.polyval(coeff, pidx)
        estbw = Config.bw * (1 + estf / Config.sig_freq)
        beta1 = betai * (1 + 2 * estf / Config.sig_freq)
        tstart = xp.polyval(coeft, pidx)
        tend = xp.polyval(coeft, pidx + 1)
        beta2 = 2 * xp.pi * (- estbw * 0.5 + estf) - tstart * 2 * beta1
        coef2d_est2 = xp.array([to_scalar(beta1), to_scalar(beta2), 0])

        # align 3rd parameter of coef2d_est2 to observed phase at tstart
        nsymbr_start = math.ceil(tstart * Config.fs + Config.nsamp / 8)
        nsymbr_end = math.ceil(tend * Config.fs - Config.nsamp / 8)
        nsymbr = xp.arange(math.ceil(tstart * Config.fs + Config.nsamp / 8), math.ceil(tend * Config.fs - Config.nsamp / 8))
        tsymbr = nsymbr / Config.fs

        sig0 = reader.get(nsymbr_start, nsymbr_end - nsymbr_start)
        sig1 = sig0 * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        data0 = myfft(sig1, n=Config.fft_n, plan=Config.plan)
        freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
        freq, valnew = optimize_1dfreq_fast(sig1, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # freqf, valnew = optimize_1dfreq(sig1, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # print(f"Initial freq offset: {freq1}, after FFT fit: {freq}, after precise fit: {freqf}, diff: {freqf - freq}")

        # adjust coef2d_est2[1] according to freq difference
        coef2d_est2[1] = 2 * xp.pi * (- estbw * 0.5 + estf + freq) - tstart * 2 * beta1
        sig2 = sig0 * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))
        # freq, valnew = optimize_1dfreq(sig2, tsymbr, freq1, Config.fs / Config.fft_n * 5)
        # logger.warning(f"{freq=} should be zero {valnew=}")
        coef2d_est2[2] += xp.angle(sig0.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))))
        # print(f"{xp.angle(sig0.dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr))))} should be zero")
        coeflist.append(coef2d_est2)
    return xp.array(coeflist)


def fitcoef4(coeff: xp.array, coeft: xp.array, reader: SlidingComplex64Reader):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi # frequency slope to phase 2d slope, *pi
    coeflist = fitcoef2(coeff, coeft, reader)
    
    # plot phase difference between consecutive symbols
    phasedifflist = xp.zeros((Config.preamble_len - 1,), dtype=xp.float32)
    for pidx in range(Config.preamble_len - 1):
        tjump = xp.polyval(coeft, pidx + 1)
        phasediff = wrap(xp.polyval(coeflist[pidx + 1], tjump) - xp.polyval(coeflist[pidx], tjump))
        phasedifflist[pidx] = phasediff
    phasedifflist_unwrap = xp.unwrap(xp.array(phasedifflist))
    # pltfig1(range(Config.preamble_len - 1), phasedifflist_unwrap, title="plot phase difference between consecutive symbols").show()

    # fit a line to phase difference to estimate cfo and time drift
    tdifflist = xp.zeros_like(phasedifflist_unwrap)
    for pidx in range(Config.preamble_len - 1):
        estbw = Config.bw * (1 + xp.polyval(coeff, pidx + 0.5) / Config.sig_freq)
        tdifflist[pidx] = phasedifflist_unwrap[pidx] / 2 / xp.pi / estbw # phasediff is caused by mismatched symbol change time, -> bw mismatch. phase = 2pi * bw * dt
    xrange = xp.arange(50, len(tdifflist)) # !!! ignore first 50 points
    tdiff_coef = xp.polyfit(xrange, tdifflist[xrange], 1)
    coeft_new = coeft.copy()
    coeft_new[-2:] += tdiff_coef
    logger.warning(f"{tdiff_coef=} {coeft=} {coeft_new=} cfo ppm from time: {1 - coeft_new[0] / Config.nsampf * Config.fs} cfo: {(1 - coeft_new[0] / Config.nsampf * Config.fs) * Config.sig_freq} unwrapped phasediff so t may shift by margin {1/Config.bw}")

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
    # coeft = fitcoef4(coeff, coeft, reader) # do this to check correctness of fitcoef4

    coeflist = fitcoef2(coeff, coeft, reader)
    coeff, coeft = symbtime(coeff, coeft, reader, coeflist, nextstep=1)
    print(coeff, coeft)
