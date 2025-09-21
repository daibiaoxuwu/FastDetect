import os
from utils import *
from Config import Config
from reader import SlidingComplex64Reader
from matplotlib import pyplot as plt


def work_new(estf, estt, file_path):
    file_size = os.path.getsize(file_path)
    complex64_size = xp.dtype(xp.complex64).itemsize
    assert complex64_size == 8
    print(f"{file_path=} Size in Number of symbols: {file_size // complex64_size // Config.nsamp}")

    reader = SlidingComplex64Reader(file_path)

    tsymblen = 2 ** Config.sf / Config.bw * (1 - estf / Config.sig_freq)
    coeff = xp.array((0, estf))
    coeft = xp.array((tsymblen, estt))

    sig = reader.get(around(estt * Config.fs) - Config.nsamp, Config.nsamp * 400)
    cfosymb = xp.exp(2j * xp.pi * estf * xp.linspace(0, (len(sig) - 1) / Config.fs, num=len(sig)))
    cfosymb = cfosymb.astype(xp.complex64)

    plt.plot(to_host(xp.unwrap(xp.angle(sig))))
    plt.plot(to_host(xp.unwrap(xp.angle(cfosymb))))
    plt.plot(to_host(xp.unwrap(xp.angle(sig * xp.conj(cfosymb)))))
    plt.axvline(Config.nsamp * 240)
    plt.title("raw signal phase")
    plt.show()