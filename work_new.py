import os
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from matplotlib import pyplot as plt


def work_new(fstart, tstart, file_path):
    file_size = os.path.getsize(file_path)
    complex64_size = xp.dtype(xp.complex64).itemsize
    assert complex64_size == 8
    print(f"{file_path=} Size in Number of symbols: {file_size // complex64_size // Config.nsamp}")

    reader = SlidingComplex64Reader(file_path)
    nsamp_small_2 = 2 ** Config.sf / Config.bw * Config.fs * (1 - fstart / Config.sig_freq)

    start_pos = around(tstart - 100)
    length = 200
    pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=(tstart,)).show()

    tstart += nsamp_small_2 * 120
    start_pos = around(tstart - 100)
    print("Avg Amplitude:", xp.mean(xp.abs(reader.get(start_pos, length))))
    pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=(tstart,)).show()

    length = around(nsamp_small_2 + 200)
    sig = reader.get(start_pos, length)
    cfosymb = xp.exp(2j * xp.pi * fstart * xp.linspace(0, (len(sig) - 1) / Config.fs, num=len(sig)))
    cfosymb = xp.unwrap(xp.angle(cfosymb)).astype(xp.float32)
    sig = xp.unwrap(xp.angle(sig))
    cfosymb += sig[100] - cfosymb[100]
    fig = pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(sig), addvline=(tstart, tstart + nsamp_small_2))
    pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(cfosymb), fig=fig).show()

    length = 200
    tstart += nsamp_small_2 * 120
    start_pos = around(tstart - 100)
    pltfig1(to_host(xp.arange(start_pos, start_pos + length)), to_host(xp.unwrap(xp.angle(reader.get(start_pos, length)))), addvline=(tstart,)).show()

    tstart -= nsamp_small_2 * 240
    
    tsymblen = 2 ** Config.sf / Config.bw * (1 - fstart / Config.sig_freq)
    coeff = xp.array((0, fstart))
    coeft = xp.array((tsymblen, tstart))

