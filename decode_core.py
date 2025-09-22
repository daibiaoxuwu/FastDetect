from Config import Config
import math
from utils import xp, to_scalar, around, pltfig1, optimize_1dfreq_fast, myfft

def decode_core(reader, tstart, tend, coeff_new, startphase, pidx):
    x1 = math.ceil(tstart * Config.fs)
    x2 = math.ceil(tend * Config.fs)
    nsymbr = xp.arange(x1, x2)
    tsymbr = nsymbr / Config.fs

    # pltfig1(tsymbr, xp.unwrap(xp.angle(reader.get(x1, x2-x1))), title=f"{pidx=}").show()
    # pltfig1(tsymbr, xp.abs(reader.get(x1, x2-x1)), title=f"{pidx=}").show()
    assert xp.mean(xp.abs(reader.get(x1, x2-x1))) > 0.1, f"{pidx=} {xp.mean(xp.abs(reader.get(x1, x2-x1)))=} too small. is symbol ending?"
    estcoef_this = xp.polyval(coeff_new, pidx)

    beta1 = Config.bw / ((2 ** Config.sf) / Config.bw) * xp.pi * (1 + 2 * estcoef_this / Config.sig_freq)
    estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
    beta2 = 2 * xp.pi * (xp.polyval(coeff_new, pidx) - estbw / 2) - tstart * 2 * beta1  # 2ax+b=differential b=differential - 2 * beta1 * time
    coef2d_est = xp.array([beta1.get(), beta2.get(), 0])

    sig2 = reader.get(x1, x2-x1) * xp.exp(-1j * xp.polyval(coef2d_est, tsymbr))
    data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
    freq1 = xp.fft.fftshift(xp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[xp.argmax(xp.abs(data0))]
    freq, valnew = optimize_1dfreq_fast(sig2, tsymbr, freq1, Config.fs / Config.fft_n * 5) # valnew may be as low as 0.3, only half the power will be collected
    # freq = freq1 # todo !!!
    # assert valnew > 0.3, f"{freq=} {freq1=} {valnew=}"
    if freq < 0: freq += estbw
    codex = freq / estbw * 2 ** Config.sf
    code = around(codex)
    print(f"{codex=} {code=}")

    tmid = tstart * (code / 2 ** Config.sf) + tend * (1 - code / 2 ** Config.sf)
    tmid = tmid.item()
    x3 = math.ceil(tmid * Config.fs)

    nsymbr1 = xp.arange(x1, x3)
    tsymbr1 = nsymbr1 / Config.fs
    nsymbr2 = xp.arange(x3, x2)
    tsymbr2 = nsymbr2 / Config.fs

    beta2 = (2 * xp.pi * (xp.polyval(coeff_new, pidx) + estbw * (code / 2 ** Config.sf - 0.5))
             - tstart * 2 * beta1)
    coef2d_est2 = xp.array([beta1.get(), beta2.get(), 0])
    coef2d_est2_2d = xp.polyval(coef2d_est2, tstart) - startphase
    coef2d_est2[2] -= coef2d_est2_2d

    beta2a = (2 * xp.pi * (xp.polyval(coeff_new, pidx) + estbw * (code / 2 ** Config.sf - 1.5))
              - tstart * 2 * beta1)
    coef2d_est2a = xp.array([beta1.get(), beta2a.get(), 0])
    coef2d_est2a_2d = xp.polyval(coef2d_est2a, tmid) - xp.polyval(coef2d_est2, tmid)
    coef2d_est2a[2] -= coef2d_est2a_2d

    res2 = reader.get(x1, x3-x1).dot(xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr1))) / xp.sum(xp.abs(reader.get(x1, x3-x1)))
    res2a = reader.get(x3, x2-x3).dot(xp.exp(-1j * xp.polyval(coef2d_est2a, tsymbr2))) / xp.sum(xp.abs(reader.get(x3, x2-x3)))

    if not (xp.abs(res2).item() > 0.7 or code > 2 ** Config.sf * 0.7) or not (xp.abs(res2a).item() > 0.7 or code < 2 ** Config.sf * 0.2):
        pltfig1(tsymbr1, xp.angle(reader.get(x1, x3-x1) * xp.exp(-1j * xp.polyval(coef2d_est2, tsymbr1))), title=f"{pidx=} 1st angle {codex=} pow={xp.abs(res2).item()}").show()
        pltfig1(tsymbr2, xp.angle(reader.get(x3, x2-x3) * xp.exp(-1j * xp.polyval(coef2d_est2a, tsymbr2))), title=f"{pidx=} 2st angle {codex=} pow={xp.abs(res2a).item()}").show()

    assert xp.abs(res2).item() > 0.7 or code > 2 ** Config.sf * 0.7, f"{pidx=} {code=} 1st power {xp.abs(res2).item()}<0.7"
    assert xp.abs(res2a).item() > 0.7 or code < 2 ** Config.sf * 0.2, f"{pidx=} {code=} 2nd power {xp.abs(res2a).item()}<0.7"

    endphase = xp.polyval(coef2d_est2a, tend)
    ifreq1 = 2 * xp.pi * (xp.polyval(coeff_new, pidx) + estbw * (code / 2 ** Config.sf - 0.5))
    ifreq2 = 2 * xp.pi * (xp.polyval(coeff_new, pidx) + estbw * (code / 2 ** Config.sf - 1.5))
    return code, endphase, coef2d_est2, coef2d_est2a, res2, res2a, ifreq1, ifreq2