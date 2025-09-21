from utils import xp, xfft, around, USE_GPU, to_host, to_scalar
from Config import Config
from matplotlib import pyplot as plt

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
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs * (1 - fstart / Config.sig_freq)
    start_pos_all = nsamp_small * pidx + tstart
    start_pos = around(start_pos_all)
    start_pos_d = start_pos_all - start_pos

    sig1 = reader.get(start_pos, Config.nsamp)
    sig2 = sig1 * refchirp
    freqdiff = start_pos_d / nsamp_small * Config.bw 

    sig2 = add_freq(sig2, freqdiff)
    return myfft(sig2, n=Config.fft_n, plan=Config.plan)

def moving_window_max(arr: xp.ndarray, half_window: int = 200) -> xp.ndarray:
    """
    Computes the moving window maximum for a 1D CuPy array.

    For each point in the array, this function finds the maximum value
    within a window of [-half_window, +half_window]. The total window
    size is (2 * half_window + 1).

    Args:
        arr (xp.ndarray): The input 1D CuPy array.
        half_window (int): The number of points on either side of the
                           current point to include in the window.

    Returns:
        xp.ndarray: A new CuPy array containing the moving window maximums.
    """
    # Pad the array to handle edge cases where the window is not full.
    # We pad with a very small number (negative infinity) so it doesn't
    # affect the maximum calculation.
    assert xp.dtype(arr) == xp.float32, f"Expected float32, got {xp.dtype(arr)}"
    padded_arr = xp.pad(arr, half_window, mode='constant', constant_values=-xp.inf)
    
    # Get the number of elements in the padded array
    n = padded_arr.size
    
    # Calculate the total window size
    window_size = 2 * half_window + 1
    
    # Create a view of the padded array using stride tricks. This creates
    # a new array that represents all the sliding windows without copying data.
    # The shape is (number_of_windows, window_size)
    # The strides are (element_stride, element_stride)
    windows = xp.lib.stride_tricks.as_strided(
        padded_arr,
        shape=(n - window_size + 1, window_size),
        strides=(padded_arr.strides[0], padded_arr.strides[0])
    )
    
    # Compute the maximum value along the last axis (axis=1) of the
    # newly created windows view. This is the core operation.
    result = xp.max(windows, axis=1)
    
    return result

def updown_gen(max_dwin, tstart, fstart, cfg, reader, upchirp, downchirp, split, Nup, Ndown, FFT_N, start_dwin):
    retsup = xp.zeros((Nup + 1, FFT_N), dtype=xp.float32)
    for i in range(Nup + 1):
        retsup[i] = moving_window_max(xp.abs(dechirp_fft(tstart, fstart, reader, downchirp, i + start_dwin, True)))

    retsdown = xp.zeros((Ndown + 1, FFT_N), dtype=xp.float32)
    for i in range(Ndown + 1):
        retsdown[i] = moving_window_max(xp.abs(dechirp_fft(tstart, fstart, reader, upchirp, i + start_dwin + cfg.sfdpos, False)))

    pair_up = retsup[:-1, :-split] + retsup[1:, split:]
    add_up = xp.sum(pair_up, axis=0)

    pair_down = retsdown[:-1, split:] + retsdown[1:, :-split]
    add_down = xp.sum(pair_down, axis=0)

    yield start_dwin, add_up, add_down, [xp.sum(xp.max(pair_up)), xp.argmax(pair_up, axis=1), 0]

    for dwin in range(start_dwin + 1, max_dwin):
        # --- Update UP-CHIRP ---
        # Shift buffer: discard the oldest (index 0), move everything up.
        retsup[:-1] = retsup[1:]
        # Compute only the single NEW FFT required for this window.
        new_up_fft = moving_window_max(xp.abs(dechirp_fft(tstart, fstart, reader, downchirp, Nup + dwin, True)))
        # if dwin == 550 or dwin == 600:
            # print(f"Debug: dwin={dwin}, new_up_fft max={to_scalar(xp.max(new_up_fft))}, argmax={to_scalar(xp.argmax(new_up_fft))}")
            # plt.plot(to_host(new_up_fft[xp.argmax(new_up_fft)-1000:xp.argmax(new_up_fft)+1000]))
            # plt.show()
        # Add the new FFT to the end of the buffer.
        retsup[Nup] = new_up_fft
        # Recalculate sums efficiently from the updated buffer.
        pair_up = xp.abs(retsup[:-1, :-split]) + xp.abs(retsup[1:, split:])
        add_up = xp.sum(pair_up, axis=0)

        # if dwin >= 450 and dwin <= 480: 
            # print(f"Debug: dwin={dwin}, add_up max={to_scalar(xp.max(add_up))}, argmax={to_scalar(xp.argmax(add_up))}")

        # --- Update DOWN-CHIRP ---
        # Shift buffer
        retsdown[:-1] = retsdown[1:]
        # Compute only the single NEW FFT.
        new_down_fft = moving_window_max(xp.abs(dechirp_fft(tstart, fstart, reader, upchirp, Ndown + dwin + cfg.sfdpos, False)))
        # Add the new FFT to the end of the buffer.
        retsdown[Ndown] = new_down_fft
        # Recalculate sums
        pair_down = xp.abs(retsdown[:-1, split:]) + xp.abs(retsdown[1:, :-split])
        add_down = xp.sum(pair_down, axis=0)

        yield dwin, add_up, add_down, [xp.abs(retsup[0, xp.argmax(add_up)]), xp.argmax(pair_up, axis=1), xp.abs(retsup[1, xp.argmax(add_up) + split])]


def detect_slow(cfg, xp, fstart, tstart, dwin, updown_generator, bwnew2, FFT_N):
    lo    = int(xp.around((-cfg.bw - cfg.cfo_range) * FFT_N / cfg.fs)) + FFT_N // 2
    hi    = int(xp.around(cfg.cfo_range * FFT_N / cfg.fs)) + FFT_N // 2

    dwin_gen, add_up, add_down, temp_result = next(updown_generator)
    assert dwin_gen == dwin    

    # --- peak search in restricted band ---
    fup   = int(to_scalar(xp.argmax(add_up[lo:hi]))) + lo
    fdown = int(to_scalar(xp.argmax(add_down[lo:hi]))) + lo

    fft_up   = (fup   - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fft_down = (fdown - FFT_N // 2) / (bwnew2 / cfg.fs * FFT_N) + 0.5
    fu = fft_up + 0.5
    fd = fft_down - 0.5

    f01 = (fu + fd) / 2
    f0  = (f01 + 0.5) % 1 - 0.5
    t0  = (f0 - fu)
    t0  = t0 % 1

    est_cfo_f = f0 * cfg.bw + fstart
    est_to_s  = (t0 + dwin) * cfg.tsig + tstart 

    ret1 = float(to_scalar(xp.max(add_up[lo:hi])))
    ret2 = float(to_scalar(xp.max(add_down[lo:hi])))
    retval = ret1 + ret2
    return retval, float(est_cfo_f), float(est_to_s), ret1, ret2, temp_result
