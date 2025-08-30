from utils import xp, xfft, around, USE_GPU
from Config import Config

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

# ---------- fixed FastDetectContext ----------

class FastDetectContext:
    def __init__(self, cfg, xp_mod, fstart):
        self.cfg = cfg
        self.xp = xp_mod
        self.fstart = fstart

        # ---- precompute chirps/consts (once) ----
        x = self.xp.arange(cfg.nsamp) * (1 + fstart / cfg.sig_freq)
        self.bwnew  = cfg.bw * (1 + fstart / cfg.sig_freq)
        self.bwnew2 = cfg.bw * (1 - 2 * fstart / cfg.sig_freq)
        beta        = cfg.bw / ((2 ** cfg.sf) / cfg.bw)
        betanew     = beta * (1 + 2 * fstart / cfg.sig_freq)

        self.upchirp   = self.xp.exp(2j * self.xp.pi * (betanew / 2 * x ** 2 / cfg.fs ** 2 + (-self.bwnew / 2) * x / cfg.fs))
        self.downchirp = self.xp.conj(self.upchirp)

        self.Nup   = cfg.preamble_len
        self.Ndown = 2
        self.FFT_N = cfg.fft_n
        self.split = int(self.xp.around(self.bwnew2 / cfg.fs * self.FFT_N))
        self.lo    = int(self.xp.around((-cfg.bw - cfg.cfo_range) * self.FFT_N / cfg.fs)) + self.FFT_N // 2
        self.hi    = int(self.xp.around(cfg.cfo_range * self.FFT_N / cfg.fs)) + self.FFT_N // 2

        # ---- allocate sliding buffers (filled in warm_start) ----
        self.up_fft_result   = self.xp.empty((2, self.FFT_N), dtype=self.xp.complex64)
        self.down_fft_result = self.xp.empty((2, self.FFT_N), dtype=self.xp.complex64)
        self.pair_up   = self.xp.zeros((self.Nup,   self.FFT_N - self.split), dtype=self.xp.float32)
        self.pair_down = self.xp.zeros((self.Ndown, self.FFT_N - self.split), dtype=self.xp.float32)
        self.add_up    = None
        self.add_down  = None
        self._warm_ok  = False

    # --- GPU/CPU helpers so we don't need external to_host/to_scalar ---
    @staticmethod
    def _to_host(a):
        return a.get() if USE_GPU else a

    @classmethod
    def _to_scalar(cls, a):
        v = cls._to_host(a)
        return float(v.item() if hasattr(v, "item") else v)

    # ---- do the initial fills for the sliding windows ----
    def warm_start(self, tstart, reader):
        # Fill UP: compute Nup-1 FFTs, build pair_up rows [0..Nup-2)
        for i in range(self.Nup - 1):
            ret = dechirp_fft(tstart, self.fstart, reader, self.downchirp, i, True)
            if ret is None:
                self._warm_ok = False
                return False
            self.up_fft_result[i % 2, :] = ret
            if i > 0:
                self.pair_up[i - 1] = self.xp.abs(self.up_fft_result[(i - 1) % 2, : -self.split]) + \
                                      self.xp.abs(self.up_fft_result[i % 2,      self.split :])

        self.add_up = self.xp.sum(self.pair_up, axis=0)

        # Fill DOWN: compute Ndown-1 FFTs (i.e. 1), build pair_down row 0
        for i in range(self.Ndown - 1):
            ret = dechirp_fft(tstart, self.fstart, reader, self.upchirp, i + self.cfg.sfdpos, False)
            if ret is None:
                self._warm_ok = False
                return False
            self.down_fft_result[i % 2, :] = ret
            if i > 0:
                # (won't execute for Ndown==2; kept for consistency)
                self.pair_down[i - 1] = self.xp.abs(self.down_fft_result[(i - 1) % 2, self.split :]) + \
                                        self.xp.abs(self.down_fft_result[i % 2,      : -self.split])

        self.add_down = self.xp.sum(self.pair_down, axis=0)
        self._warm_ok = True
        return True

    # ---- slide one step and compute detection for window index dwin ----
    def run(self, tstart, reader):
        if not self._warm_ok:
            ok = self.warm_start(tstart, reader)
            if not ok:
                return None, None, None, None, None

        # --- advance UP window ---
        i = self.Nup - 1
        ret = dechirp_fft(tstart, self.fstart, reader, self.downchirp, i, True)
        if ret is None: return None, None, None, None, None
        self.up_fft_result[i % 2, :] = ret

        # remove oldest row, add new row
        oldest_row = i % self.Nup
        self.add_up -= self.pair_up[oldest_row, :]
        self.pair_up[oldest_row, :] = \
            self.xp.abs(self.up_fft_result[(i - 1) % 2, : -self.split]) + \
            self.xp.abs(self.up_fft_result[i % 2,      self.split  :])
        self.add_up += self.pair_up[oldest_row, :]

        # --- advance DOWN window ---
        i2 = self.Ndown - 1
        ret = dechirp_fft(tstart, self.fstart, reader, self.upchirp, i2 + self.cfg.sfdpos, False)
        if ret is None: return None, None, None, None, None
        self.down_fft_result[i2 % 2, :] = ret

        oldest_row_d = i2 % self.Ndown
        self.add_down -= self.pair_down[oldest_row_d, :]
        self.pair_down[oldest_row_d, :] = \
            self.xp.abs(self.down_fft_result[(i2 - 1) % 2, self.split :]) + \
            self.xp.abs(self.down_fft_result[i2 % 2,       : -self.split])
        self.add_down += self.pair_down[oldest_row_d, :]

        # --- peak search in restricted band ---
        lo, hi, FFT_N = self.lo, self.hi, self.FFT_N
        fup   = int(self._to_scalar(self.xp.argmax(self.add_up[lo:hi]))) + lo
        fu_sec = -1 if fup > -self.cfg.bw // 2 * FFT_N / self.cfg.fs + FFT_N // 2 else 1

        fdown = int(self._to_scalar(self.xp.argmax(self.add_down[lo:hi]))) + lo
        fd_sec = -1 if fdown > FFT_N // 2 else 1

        # --- CFO/TO grid (same maths as your code) ---
        fft_up   = (fup   - FFT_N // 2) / (self.bwnew2 / self.cfg.fs * FFT_N) + 0.5
        fft_down = (fdown - FFT_N // 2) / (self.bwnew2 / self.cfg.fs * FFT_N) + 0.5
        fu = fft_up
        fd = fft_down

        f01 = (fu + fd) / 2
        f0  = (f01 + 0.5) % 1 - 0.5
        t0  = (f0 - fu - 0.5)
        t0  = t0 % 1 - 1

        est_cfo_f = f0 * self.cfg.bw + self.fstart
        est_to_s  = t0 * self.cfg.tsig + tstart 

        ret1 = float(self._to_scalar(self.xp.max(self.add_up[lo:hi])))
        ret2 = float(self._to_scalar(self.xp.max(self.add_down[lo:hi])))
        retval = ret1 + ret2
        return retval, float(est_cfo_f), float(est_to_s), ret1, ret2

    __call__ = run  # so you can call ctx(tstart, reader, dwin)
