from functools import lru_cache
from utils import xp, around 
import numpy as np


def symbol_generation_by_frequency(encoded, SF, BW, Fs):
    coeff_vec = (1, 0)
    """
    生成一个符号（非线性 LoRa 版本）
    -------------------------------------------------
    功能：
      - 允许使用 coeff_vec 控制扫频曲线的形状
      - 支持标准线性 chirp (coeff_vec=(1,)) 和非线性 chirp
      - 频率始终限制在 [-BW/2, +BW/2] 区间（带宽内 wrap-around）

    参数：
      encoded    : 符号索引 (0 ~ 2^SF - 1)
      SF         : 扩频因子 (Spreading Factor)
      coeff_vec  : 多项式系数向量 (控制频率随时间的变化)
                   例如 (1,) 表示线性，(0.5,0.5) 表示二次弯曲
      BW         : 带宽 (Hz)
      Fs         : 采样率 (Hz)

    返回：
      y : 复基带信号，一个符号时长的 chirp (长度约 T*Fs)
    """
    # === 符号时长 & 采样时间 ===
    T = (2 ** SF) / BW                          # 单个符号时长 (秒)
    n = int(np.floor(T * Fs)) + 1               # 采样点数 (+1 确保覆盖末尾)
    t = np.arange(n) / Fs                       # 时间向量 [0, 1/Fs, 2/Fs, ..., T]

    # === 处理多项式系数 ===
    a = np.array(coeff_vec, dtype=float).ravel()  # 转成 numpy 向量
    s = a.sum()
    if s == 0:
        raise ValueError("coefficient_vector sum = 0")  # 防止全零
    a = a / s                                        # 归一化，保证总权重为 1

    # === 构造扫频多项式 ===
    degree = len(a)                                  # 多项式阶数
    divisors = T ** (degree + 1 - np.arange(1, degree + 1))
    freq_coeff = BW * (a / divisors)                 # 缩放到带宽范围

    # === 加入常数项 & 符号偏移 ===
    # -BW/2 : 保证频率区间对称 [-BW/2, BW/2]
    # encoded/T : 符号的起始频率偏移 (决定符号索引)
    coeff = np.concatenate([freq_coeff, [-BW / 2.0 + encoded / T]])

    # === 计算瞬时频率 (带 wrap-around) ===
    # np.polyval(coeff, t) : 多项式形式的扫频轨迹
    # wrap 到 [-BW/2, BW/2]
    freq = (np.polyval(coeff, t) + BW / 2) % BW - BW / 2

    # === 积分得到相位 ===
    phase = 2 * np.pi * np.cumsum(freq) / Fs

    # === 生成复基带信号 ===
    y = np.exp(1j * phase)


    T = (2**SF) / BW

    # 2. Calculate the term inside the square root
    # This is derived from t^2 = T^2 - (T * encoded) / BW
    inner_term = T**2 - (T * encoded) / BW

    t = np.sqrt(inner_term)

    # === 返回结果 ===
    return y[:-1], int(np.ceil(t * Fs))  # 去掉最后一个采样点，保证符号长度一致

@lru_cache(maxsize=None)
def build_decode_matrices(n_classes: int, nsamp: int, fs: float, bw: float, sf: int, is_curving: bool):
    t = xp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    A = xp.zeros((n_classes, nsamp), dtype=xp.complex64)
    B = xp.zeros((n_classes, nsamp), dtype=xp.complex64)

    betai = bw / ((2 ** sf) / bw)
    wflag = True
    for code in range(n_classes):
        if (code - 1) % 4 != 0 and sf >= 11 and wflag:
            wflag = False
            continue
        if is_curving:
            symb, tjump = symbol_generation_by_frequency(code, sf, bw, fs)
            symb = xp.conj(xp.array(symb))
            assert tjump > 0 and tjump <= nsamp 
            A[code, :tjump] = xp.array(symb[:tjump], dtype=xp.complex64)
            if code > 0:
                B[code, tjump:] = xp.array(symb[tjump:], dtype=xp.complex64)
        else:
            nsamples = around(nsamp / n_classes * (n_classes - code))
            f01 = bw * (-0.5 + code / n_classes)
            ref1 = xp.exp(-1j * 2 * xp.pi * (f01 * t + 0.5 * betai * t * t))
            f02 = bw * (-1.5 + code / n_classes)
            ref2 = xp.exp(-1j * 2 * xp.pi * (f02 * t + 0.5 * betai * t * t))
            A[code, :nsamples] = ref1[:nsamples]
            if code > 0:
                B[code, nsamples:] = ref2[nsamples:]
    return A, B, t

def decode_payload(reader, Config, est_cfo_f: float, est_to_s: float, is_curving: bool):
    A, B, t = build_decode_matrices(Config.n_classes, Config.nsamp, Config.fs, Config.bw, Config.sf, is_curving)

    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * (1 + 2 * est_cfo_f / Config.sig_freq)
    codes = []
    phasediffs = []

    for pidx in range(Config.sfdend, Config.total_len):
        start_pos_all = (2 ** Config.sf / Config.bw) * Config.fs * (pidx + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = around(start_pos_all)
        dt = (start_pos - start_pos_all) / Config.fs

        dataX = reader.get(start_pos, Config.nsamp) * xp.exp(-1j * 2 * xp.pi * (est_cfo_f + betai * dt) * t)
        data1 = A @ dataX
        data2 = B @ dataX
        vals = xp.abs(data1) ** 2 + xp.abs(data2) ** 2
        coderet = int(xp.argmax(vals).item())
        codes.append(coderet)
        phasediff = (xp.angle(data1[coderet]) - xp.angle(data2[coderet])) % (2 * xp.pi)
        phasediffs.append(phasediff.item())
    print(phasediffs)
    return codes
