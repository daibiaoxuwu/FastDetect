import plotly.graph_objects as go
import logging
import os

# utils.py
import os

USE_GPU = os.getenv("USE_GPU", "0") == "1"

if USE_GPU:
    try:
        import cupy as xp
        import cupyx.scipy.fft as xfft
        on_gpu = True

        asnumpy = xp.asnumpy    # CuPy → NumPy
        asarray = xp.asarray    # Python/NumPy → CuPy
    except ImportError:
        # fallback if cupy not installed
        import numpy as xp
        import scipy.fft as xfft
        on_gpu = False

        asnumpy = lambda a: a
        asarray = xp.asarray
else:
    import numpy as xp
    import scipy.fft as xfft
    on_gpu = False

    asnumpy = lambda a: a
    asarray = xp.asarray

logging.basicConfig( format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.INFO )
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(f'run_250828.log')
file_handler.setLevel(level=logging.INFO)
logger.addHandler(file_handler)

# -------- light helpers (backend-agnostic) --------
def to_device(x):
    """Move/convert Python or NumPy data to the active backend array (CuPy on GPU, NumPy on CPU)."""
    return asarray(x)

def to_host(x):
    """Ensure a NumPy array on host memory (identity on CPU)."""
    return asnumpy(x)

def to_scalar(x):
    """Return a Python scalar if x is 0-d array/array-like; otherwise return x unchanged."""
    # Works for both NumPy and CuPy
    try:
        if getattr(x, "shape", None) == ():
            return x.item()
    except Exception:
        pass
    return x

def to_scalar_list(lst):
    """Map list of numbers/0-d arrays to pure Python scalars."""
    out = []
    for v in lst:
        try:
            if getattr(v, "shape", None) == ():
                out.append(v.item())
            else:
                out.append(v)
        except Exception:
            out.append(v)
    return out

def around(x):
    """Round to nearest integer (works for Python num or 0-d array)."""
    return round(float(to_scalar(x)))



def pltfig(datas, title = None, yaxisrange = None, modes = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    datas : list of tuples
        Each tuple contains two lists or array-like elements, the data for the x and y axes.
        If only y data is provided, x data will be generated using xp.arange.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if not all(len(data) == 2 for data in datas): datas = [(xp.arange(len(data)), data) for data in datas]
    if modes is None:
        modes = ['lines' for _ in datas]
    elif isinstance(modes, str):
        modes = [modes for _ in datas]
    for idx, ((xdata, ydata), mode) in enumerate(zip(datas, modes)):
        if line == None and idx == 1: line = dict(dash='dash')
        fig.add_trace(go.Scatter(x=to_scalar_list(xdata), y=to_scalar_list(ydata), mode=mode, marker=marker, line=line))
        assert len(to_scalar_list(xdata)) == len(to_scalar_list(ydata))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig



def pltfig1(xdata, ydata, title = None, yaxisrange = None, mode = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    xdata : list or array-like or None
        The data for the x-axis.
        If is None, and only y data is provided, x data will be generated using xp.arange.
    ydata : list or array-like
        The data for the y-axis.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if xdata is None: xdata = xp.arange(len(ydata))
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if mode is None: mode = 'lines'
    fig.add_trace(go.Scatter(x=to_scalar_list(xdata), y=to_scalar_list(ydata), mode=mode, marker=marker, line=line))
    assert len(to_scalar_list(xdata)) == len(to_scalar_list(ydata))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig

def pltfig_hind(addhline, addvline, line_dash_in, fig, yaxisrange):
    if yaxisrange: fig.update_layout(yaxis=dict(range=yaxisrange), )
    if addvline is not None:
        if line_dash_in is None:
            line_dash = ['dash' for _ in range(len(addvline))]
            line_dash[0] = 'dot'
        elif isinstance(line_dash_in, str):
            line_dash = [line_dash_in for _ in range(len(addvline))]
        else: line_dash = line_dash_in
        for x, ldash in zip(addvline, line_dash): fig.add_vline(x=to_scalar(x), line_dash=ldash)
    if addhline is not None:
        if line_dash_in is None:
            line_dash = ['dash' for _ in range(len(addhline))]
            line_dash[0] = 'dot'
        elif isinstance(line_dash_in, str):
            line_dash = [line_dash_in for _ in range(len(addhline))]
        else: line_dash = line_dash_in
        for y, ldash in zip(addhline, line_dash): fig.add_hline(y=to_scalar(y), line_dash=ldash)
