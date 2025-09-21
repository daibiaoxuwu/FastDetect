import numpy as np
import cupy as cp
import plotly.graph_objects as go
from math import ceil, floor
from Config import Config
from utils import to_host, to_device, sqlist, wrap, around, pltfig, pltfig1, to_scalar

# Define the coefficients for the two polynomials
coeflist = [
    [1.26581062e+08, -6.65625087e+06, 8.42679006e+04],
    [1.26581062e+08, -9.20875231e+06, 1.64245710e+05]
]


# Define the start position and the range around it
tstart2 = 0.030318928843199852
epsilon = 1e-6

# Function to find crossing points where poly(x) = (2n +1)*pi
def find_crossings(coef_in, x_min, x_max):
    coef = coef_in.copy()
    # coef[2] %= (2 * cp.pi)
    # ymin ymax: suppose is increasing/decreasing in this section
    assert not x_min <= - coef[1] / 2 / coef[0] <= x_max

    ya = cp.polyval(coef, x_min)
    yb = cp.polyval(coef, x_max)
    y_lower = min(ya, yb)
    y_upper = max(ya, yb)

    # Compute the range of n values
    n_min = ceil((y_lower - cp.pi) / (2 * cp.pi))
    n_max = floor((y_upper - cp.pi) / (2 * cp.pi))

    crossings = []
    nrange = np.arange(int(n_min), int(n_max) + 1)
    nrangex = np.arange(int(n_min), int(n_max) + 2)

    if coef[0] * 2 * x_min + coef[1] < 0: # decreasing
        assert coef[0] * 2 * x_max + coef[1] < 0
        nrange = nrange[::-1]
        nrangex = nrangex[::-1]

    for n in nrange:
        # Solve poly(x) - (2n +1)*pi = 0
        shifted_poly = coef.copy()
        shifted_poly[2] -= (2 * n + 1) * cp.pi
        roots = to_device(np.roots(to_host(shifted_poly)))
        # Filter real roots within [x_min, x_max]
        real_roots = roots[cp.isreal(roots)].real
        valid_roots = real_roots[(real_roots >= x_min) & (real_roots <= x_max)]
        assert len(valid_roots) == 1 or len(valid_roots) == 2, f"valid roots {shifted_poly=} {roots=} {x_min=} {x_max=} {y_lower=} {y_upper=} {n_min=} {n_max=}"
        assert - np.pi <= np.polyval(shifted_poly, valid_roots[0]) <= np.pi, f"valid roots {shifted_poly=} {roots=}"
        crossings.extend(valid_roots.tolist())
    # slope: is strictly increasing or decreasing
    for i in range(len(crossings) - 1):
        xv = (crossings[i] + crossings[i + 1]) / 2
        # print( - np.pi + 2 * np.pi * nrangex[i + 1] , np.polyval(coef, xv) , np.pi + 2 * np.pi * nrangex[i + 1])
        assert - np.pi + 2 * np.pi * nrangex[i + 1] <= np.polyval(coef, xv) <= np.pi + 2 * np.pi * nrangex[i + 1]

    assert len(crossings) == len(nrangex) - 1
    # print(np.min(np.diff(crossings)) < 0, coef[0] * 2 * x_min + coef[1] < 0, coef[0] * 2 * x_max + coef[1] < 0)
    if len(crossings) > 0:
        if len(crossings) > 1: assert np.min(np.diff(crossings)) > 0
        assert np.min(crossings) > x_min, f"{np.min(crossings)=} {x_min=}"
        assert np.max(crossings) < x_max, f"{np.max(crossings)=} {x_max=}"
    return crossings, nrangex


def merge_crossings(crossings_poly1, crossing_n1, crossings_poly2, crossing_n2, x_min, x_max, tol=1e-12):
    # Combine the intersection points and sort them
    # print(crossings_poly1)
    # print(crossings_poly2)
    # print(crossing_n1)
    # print(crossing_n2)
    merged_crossings = sorted(set(crossings_poly1 + crossings_poly2))

    # Remove crossings that are too close to each other
    if len(merged_crossings) == 0: filtered_crossings = []
    else:
        filtered_crossings = [merged_crossings[0]]
        for x in merged_crossings[1:]:
            if abs(x - filtered_crossings[-1]) >= tol:
                filtered_crossings.append(x)

    # Initialize the weight list for the filtered sections
    merged_weights = []

    sections1 = [x_min, *crossings_poly1, x_max]
    sections2 = [x_min, *crossings_poly2, x_max]
    sections = [x_min, *filtered_crossings, x_max]
    sections1 = [(sections1[x], sections1[x + 1]) for x in range(len(sections1) - 1)]
    sections2 = [(sections2[x], sections2[x + 1]) for x in range(len(sections2) - 1)]
    sections = [(sections[x], sections[x + 1]) for x in range(len(sections) - 1)]

    for (x_left, x_right) in sections:
        # Determine the weights for the current section from crossing_n1 and crossing_n2
        weight1 = next((w for (x1, x2), w in zip(sections1, crossing_n1) if x1 <= x_left <= x_right <= x2), None)
        weight2 = next((w for (x1, x2), w in zip(sections2, crossing_n2) if x1 <= x_left <= x_right <= x2), None)
        # print(weight1, weight2)

        merged_weights.append((weight1, weight2))

    assert len(sections) == len(merged_weights)
    return sections, merged_weights



# Function to determine n for wrapping within a section
def determine_n(coef, x_start, x_end):
    # Choose the midpoint to determine n
    x_mid = (x_start + x_end) / 2
    y_mid = cp.polyval(coef, x_mid)
    n = floor((y_mid + cp.pi) / (2 * cp.pi))
    return n


def find_intersections(coefa, coefb, tstart2,reader, epsilon, margin=10, draw=False, remove_range = True):
    selected = None
    x_min = tstart2 - epsilon
    x_max = tstart2 + epsilon

    # Find all crossing points for both polynomials
    crossings_poly1, crossing_n1 = find_crossings(coefa, x_min, x_max)
    crossings_poly2, crossing_n2 = find_crossings(coefb, x_min, x_max)

    # Combine and sort all crossing points
    sections, all_weights = merge_crossings(crossings_poly1, crossing_n1, crossings_poly2, crossing_n2, x_min, x_max)
    # print(sections)
    # print(all_weights)
    assert len(sections) == len(all_weights)

    for i, (x1, x2) in enumerate(sections):
        xv = (x1 + x2) / 2
        # print(i, - np.pi + 2 * np.pi * all_weights[i][0] , np.polyval(coefa, xv) , np.pi + 2 * np.pi * all_weights[i][0])
        # print(i, - np.pi + 2 * np.pi * all_weights[i][1] , np.polyval(coefb, xv) , np.pi + 2 * np.pi * all_weights[i][1])
        assert - np.pi + 2 * np.pi * all_weights[i][0] <= np.polyval(coefa, xv) <= np.pi + 2 * np.pi * all_weights[i][0]
        assert - np.pi + 2 * np.pi * all_weights[i][1] <= np.polyval(coefb, xv) <= np.pi + 2 * np.pi * all_weights[i][1]

    intersection_points = []
    fig = go.Figure(layout_title_text=f"finditx {tstart2=}")
    for idx, ((x_start, x_end), (n1, n2)) in enumerate(zip(sections, all_weights)):

        # Adjust the constant term for wrapping
        # Create shifted polynomials by adding (2n -1)*pi to the constant term
        poly1_shifted_coef = coefa.copy()
        poly1_shifted_coef[2] -= (2 * n1) * cp.pi

        poly2_shifted_coef = coefb.copy()
        poly2_shifted_coef[2] -= (2 * n2) * cp.pi

        # Compute the difference polynomial
        poly_diff = cp.polysub(poly1_shifted_coef, poly2_shifted_coef)

        # Find roots of the difference polynomial
        roots = to_device(np.roots(to_host(poly_diff)))
        # Filter real roots
        real_roots = roots[cp.isreal(roots)].real
        # Filter roots within the current section
        valid_roots = real_roots[(real_roots >= x_start) & (real_roots <= x_end)]
        # Convert to float and remove duplicates
        valid_roots = valid_roots.astype(float)

        # Add valid roots to intersection points
        intersection_points.extend(valid_roots)
        # Generate x values for plotting

        if draw:
            num_points = 20
            x_vals = np.linspace(x_start, x_end, num_points)
            y1_wrapped = np.polyval(poly1_shifted_coef, x_vals)
            y2_wrapped = np.polyval(poly2_shifted_coef, x_vals)

            # Add shifted polynomials to the subplot
            fig = pltfig(((x_vals, y1_wrapped), (x_vals, y2_wrapped)), addvline=(x_start, x_end), addhline=(-np.pi, np.pi), fig = fig)
            fig.add_vline(x = to_scalar(tstart2), line=dict(color='blue'))


            # Highlight intersection points within this section
            if len(valid_roots) > 0:
                for x_int in valid_roots:
                    fig.add_trace(
                        go.Scatter(x=(to_host(x_int),), y=(np.polyval(to_host(poly1_shifted_coef), to_host(x_int)),), mode='markers', name='Intersections',
                                   marker=dict(color='red', size=8, symbol='circle')),
                    )
                    fig.add_trace(
                        go.Scatter(x=(to_host(x_int),), y=(np.polyval(to_host(poly1_shifted_coef), to_host(x_int)),), mode='markers', name='Intersections',
                                   marker=dict(color='red', size=8, symbol='circle')),
                    )
                # fig.update_yaxes(range=[-np.pi * 5 - 0.1, np.pi * 5 + 0.1])


    xv = to_device(np.arange(np.ceil(to_host(x_min) * Config.fs), np.ceil(to_host(x_max) * Config.fs), dtype=int))
    sig = reader.get(to_scalar(xv[0]), len(xv))
    val1 = cp.cos(cp.polyval(coefa, xv / Config.fs) - cp.angle(sig))
    val2 = cp.cos(cp.polyval(coefb, xv / Config.fs) - cp.angle(sig))

    if len(intersection_points) != 0:
        selected = max(intersection_points, key=lambda x: np.sum(val1[:np.ceil(x * Config.fs - xv[0])]) + np.sum(val2[np.ceil(x * Config.fs - xv[0]):]))
        selected2 = min(intersection_points, key=lambda x: abs(x - tstart2))
        if selected2 != selected:
            # logger.warning(f"find_intersections(): break point not closeset to tstart2 selected {selected - tstart2 =}")
            if remove_range: return None

    if draw:
        vals = [cp.sum(val1[:np.ceil(x * Config.fs - xv[0])]) + cp.sum(val2[np.ceil(x * Config.fs - xv[0]):]) for x in
                intersection_points]
        # pltfig1(intersection_points, vals, mode="markers", title="temp1").show()
        xv = to_device(np.arange(np.ceil(to_host(x_min) * Config.fs), np.ceil(to_host(x_max) * Config.fs), dtype=int))
        fig.add_trace(
            go.Scatter(x=to_host(xv / Config.fs), y=to_host(cp.angle(sig)), mode='markers',
                       name='rawdata',
                       marker=dict(color='blue', size=4, symbol='circle')),
        )
        if len(intersection_points) != 0:
            fig.add_vline(x = selected, line=dict(color='red'))


        fig.show()

        # a1 = []
        # x1 = []
        # for i in range(-3000, 0):
        #     xv1 = np.arange(around(tstart2 * Config.fs + i - margin), around(tstart2 * Config.fs + i + margin), dtype=int)
        #     a1v = cp.angle(pktdata_in[xv1].dot(cp.exp(-1j * cp.polyval(coefa, xv1 / Config.fs))))
        #     x1.append(around(tstart2 * Config.fs + i) )
        #     a1.append(a1v)
        # for i in range(1, 3000):
        #     xv1 = np.arange(around(tstart2 * Config.fs + i - margin), around(tstart2 * Config.fs + i + margin), dtype=int)
        #     a1v = cp.angle(pktdata_in[xv1].dot(cp.exp(-1j * cp.polyval(coefb, xv1 / Config.fs))))
        #     x1.append(around(tstart2 * Config.fs + i) )
        #     a1.append(a1v)
        # pltfig1(x1, a1, title="angle difference").show()


    # Print the intersection points
    # print("Intersection Points within the specified range:")
    for idx, x in enumerate(intersection_points, 1):
        y1 = wrap(np.polyval(to_host(coefa), to_host(x)))
        y2 = wrap(np.polyval(to_host(coefb), to_host(x)))
        assert abs(y1 - y2) < 1e-6

        # print(f"{idx}: x = {x:.12f}, y1 = {y1:.12f} y2 = {y2:.12f} y1-y2={y1-y2:.12f}")
    return selected
