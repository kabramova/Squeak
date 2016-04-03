from __future__ import division
import pandas as pd
import numpy as np
from scipy import interp, stats
from scipy.stats import ttest_ind
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')


""" Utils """

# def smooth_gaussian(array, degree=5):
#     """
#     Smoothes jagged, oversampled time series data.
#     :param array: array to be smoothed
#     :param degree: (int, default=5) smoothing window
#     Code from http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/
#     """
#     window = degree * 2 - 1
#     weight = np.array([1.0] * window)
#     weight_gauss = []
#     for i in range(window):
#         i = i - degree + 1
#         frac = i / float(window)
#         gauss = 1 / (np.exp((4 * (frac)) ** 2))
#         weight_gauss.append(gauss)
#     weight = np.array(weight_gauss) * weight
#     smoothed = [0.0] * (len(array) - window)
#     for i in range(len(smoothed)):
#         smoothed[i] = sum(np.array(array[i:i + window]) * weight) / sum(weight)
#     return smoothed
#
# def smooth(x, window_len=11, window='hanning'):
#     """Smooth the data using a window with requested size
#     http://wiki.scipy.org/Cookbook/SignalSmooth
#    """
#     if x.ndim != 1:
#         raise (ValueError, "smooth only accepts 1 dimension arrays.")
#     if x.size < window_len:
#         raise (ValueError, "Input vector needs to be bigger than window size.")
#     if window_len < 3:
#         return x
#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
#     s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
#     # print(len(s))
#     if window == 'flat':  # moving average
#         w = np.ones(window_len, 'd')
#     else:
#         w = eval('np.' + window + '(window_len)')
#     y = np.convolve(w / w.sum(), s, mode='valid')
#     return y
#
#
# def smooth_timeseries(series, window_len=11, window='hanning'):
#     original_index = series.index
#     smoothed = smooth(series, window_len, window)
#     # Return to original length via interpolation
#     interpolated = np.interp(np.linspace(0, len(smoothed), len(original_index)),
#                              range(len(smoothed)), smoothed)
#     return pd.TimeSeries(interpolated, original_index)
#
#
# def map_bin(x, bins):
#     """Bin data"""
#     kwargs = {}
#     if x == max(bins):
#         kwargs['right'] = True
#         bin = bins[np.digitize([x], bins, **kwargs)[0]]
#     bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
#     return bin_lower
#
#
# def bin_series(series, bins=None):
#     if bins == None:
#         maximum = series.max()
#         bins = np.arange(0, maximum+.1, maximum/len(series))
#     binned = series.index.map(lambda s: map_bin(s, bins))
#     raw = pd.DataFrame(data=zip(series, binned), index=series.index, columns=['val', 'bin'])
#     grouped = raw.groupby('bin').mean()
#     return pd.TimeSeries(grouped.val, bins)


""" Normalization and preprocessing """


def list_from_string(string_list):
    """
    Parses string representation of list '[1, 2, 3]' to an actual list [1,2,3]
    :param string_list: input string
    :return: output list
    """
    first = string_list.strip('[]')
    second = np.array(first.split(','))
    third = second.astype(float)
    return third


def get_good_idx(y, threshold1, threshold2):
    """
    Get indices of when the trajectory starts, i.e. the cue appeared, and when it ends, i.e. the rest is nan-padding.
    :param y:
    :param threshold1:
    :param threshold2:
    :return:
    """
    y = y.values
    #print y
    crossed_button = np.where(y > threshold1)
    crossed_stim = np.where(y < threshold2)

    if crossed_button[0].size == 0 or crossed_stim[0].size == 0:
        idx_start = np.nan
        idx_end = np.nan
    else:
        cb_ix = crossed_button[0][0]
        cs_ix = crossed_stim[0]
        crossed_ix = np.where(cs_ix > cb_ix)
        if crossed_ix[0].size == 0:
            idx_start = np.nan
            idx_end = np.nan
        else:
            idx_start = cs_ix[crossed_ix[0][0]]
            ynans = np.where(np.isnan(y))
            if ynans[0].size == 0:
                idx_end = -1
            else:
                idx_end = ynans[0][0]
    return idx_start, idx_end


def detect_missing(ix, selected):
    return np.any(np.isnan(ix)) or ix[1]-ix[0] == 1 or selected == -1


# def get_good_coord(x, y, t, ix):
#     # print ix
#     if np.any(np.isnan(ix)):
#         good_x = np.empty(1)
#         good_y = np.empty(1)
#         good_t = np.empty(1)
#     else:
#         idx_start = ix[0]
#         idx_end = ix[1]
#         good_x = np.array(x[idx_start:idx_end])
#         good_y = np.array(y[idx_start:idx_end])
#         good_t = np.array(t[idx_start:idx_end])
#     return good_x, good_y, good_t


def get_good_coord(coord, ix):
    # print ix
    if np.any(np.isnan(ix)):
        good_coord = []
    else:
        idx_start = ix[0]
        idx_end = ix[1]
        good_coord = coord[idx_start:idx_end]
    return list(good_coord)


def get_condition(color, pos):
    if color == pos:
        return 'congruent'
    else:
        return 'incongruent'


# def normalize_x(coordinates):
#     nan_ix = np.where(np.isnan(coordinates))
#     coordinates = coordinates - coordinates[0]
#     if nan_ix[0].size == 0:
#         last_index = -1
#     else:
#         last_index = nan_ix[0][0]-1
#     if coordinates[0] > coordinates[last_index]:
#             coordinates = -1 * coordinates
#     return coordinates


def normalize_x(coordinates):
    if len(coordinates) == 0:
        new_coordinates = []
    else:
        new_coordinates = coordinates - coordinates[0]
        if new_coordinates[0] > new_coordinates[-1]:
                new_coordinates = -1 * new_coordinates
    return new_coordinates


def normalize_y(coordinates):
    if len(coordinates) == 0:
        return []
    else:
        return -1 * (coordinates - coordinates[0])


def rescale_space(coordinates, new_min, new_max, old_range):
    """
    Rescales space to given x- and y-limits and centers on the origin.
    :param coordinates: (array) array of coordinates to rescale
    :param new_min: (int) new minimum limit
    :param new_max: (int) new maximum limit
    :param old_range: (int) original range of the data (screen width or height)
    :return: space-normalized array
    """
    new_range = new_max - new_min
    old_min = 0
    rescaled = map(lambda x: ((((x - old_min) * new_range) / old_range) + new_min), coordinates)
    # rescaled = np.array(rescaled)
    return list(rescaled)


def normalize_time(coord, t, length=101):
    """
    Interpolates x/y coordinates and t to 101 even time steps.
    :param coord: (array) coordinates to be interpolated
    :param t: (array) associated timestamps
    :param length: (int, default=101) number of time steps to interpolate to
    :return: (list) interpolated x- and y-coordinates
    """
    if len(coord) < 2:
        return [np.nan] * length
    else:
        nt = np.arange(min(t), max(t), (float(max(t) - min(t)) / length))
        normalized = interp(nt, t, coord)[:101]
        return list(normalized)


# def normalize_time(x, y, t, length=101):
#     """
#     Interpolates x/y coordinates and t to 101 even time steps.
#     :param x: (array) coordinates to be interpolated
#     :param y: (array) coordinates to be interpolated
#     :param t: (array) associated timestamps
#     :param length: (int, default=101) number of time steps to interpolate to
#     :return: (Series) interpolated x- and y-coordinates
#     """
#     nt = np.arange(min(t), max(t), (float(max(t) - min(t)) / length))
#     nx = interp(nt, t, x)[:101]
#     ny = interp(nt, t, y)[:101]
#     return pd.Series(nx, range(len(nx))), pd.Series(ny, range(len(ny)))


# def average_path(coord, subject, condition):
#     """
#     Produce average path of given coordinates grouped by subject and condition.
#     :param coord: coordinates to average
#     :param subject: column with subject ids
#     :param condition: column with condition ids
#     :return:
#     """
#     # convert all coordinates to a string representation
#     df1 = coord.apply(str)
#     # strip resulting brackets
#     df2 = df1.apply(lambda trial: trial.strip('[]'))
#     # combine all rows into a list
#     df3 = pd.Series(df2.values.tolist())
#     # split the list of values into separate columns
#     df4 = df3.str.split(',', expand=True).astype(float)
#     subj = pd.DataFrame({'subj':subject.values})
#     cond = pd.DataFrame({'cond':condition.values})
#     merged = pd.concat([subj, cond, df4], axis=1)  # type: pd.DataFrame
#     grouped = merged.groupby(['cond'], as_index=False)
#     averaged = grouped.mean()
#     return merged, averaged


def average_path(data, coord, grouping):
    """
    Produce average path of given coordinates grouped by subject and condition.
    :param data: data frame
    :param coord: (str) column with coordinates
    :param grouping: (str) column with grouping condition
    :return:
    """
    # dataMidx = data.set_index('condition')
    # sLieX = dataMidx.ix['lie','nx'].mean()
    s = data.groupby(grouping).apply(lambda x: np.nanmean(x[coord]))
    return s

# mx, gx = average_path(data['nx'], data['subject_nr'], data['condition'])
# my, gy = average_path(data['ny'], data['subject_nr'], data['condition'])
# xlie = np.array(gx.iloc[0][2:]).astype(float)
# xtruth = np.array(gx.iloc[1][2:]).astype(float)
# ylie = np.array(gy.iloc[0][2:]).astype(float)
# ytruth = np.array(gy.iloc[1][2:]).astype(float)
# plt.plot(xlie, ylie)
# plt.plot(xtruth, ytruth)


""" Dependent variables """


# np.argmax(): It indices of maximum


def total_duration(t):
    return t[-1]


def get_latency(x, y, t):
    path = np.array(list(zip(x, y)))
    idx = None
    for i in range(len(path)-1):
        if not np.array_equal(path[i], path[i+1]):
            idx = i
            break
    return t[idx]


def total_inmotion(x, y, t):
    latency = get_latency(x, y, t)
    return t[-1] - latency


def get_dwell(x, y, t, dwell_size):
    xlimit = dwell_size * 1
    ylimit = dwell_size * 1.5
    path = np.array(list(zip(x, y)))
    res = path < (path[-1] - np.array([xlimit, ylimit]))
    idx = None
    for i in reversed(range(len(res))):
        if False not in res[i]:
            idx = i
            break
    return t[-1] - t[idx]


def get_init_time(t, y, y_threshold):
    """
    Returns time from t of point where y exceeds y_threshold.
    :param t: (array) time-stamps of the trajectory
    :param y: (array) y-coordinates
    :param y_threshold: (int) y-threshold
    :return: Timestamp of first y value to exceed y_threshold.
    """
    init_time = next(time for (time, location) in zip(t, y) if location < y_threshold)
    return init_time


def euc_dist(x, y):
    """
    Calculate Euclidean distances travelled by the trajectory at each time step.
    :param x: (array) x-coordinates
    :param y: (array) y-coordinates
    :return: Euclidean distance
    """
    xdist = np.ediff1d(x)**2
    ydist = np.ediff1d(y)**2
    dist = np.sqrt(xdist+ydist)
    return dist


def dist_from_response(x, y, foil=False):
    """
    Calculate distance from the ultimate response for each step of a trajectory.
    :param x: (array) x-coordinates
    :param y: (array) y-coordinates
    :param foil: (bool, default=False) if True, shows distance from the foil response.
    :return: (array) distances
    """
    # TODO make these reference targets flexible as input
    # Infer response locations
    response_x, response_y = (x[-1], y[-1])
    if foil:
        response_x *= -1
    # Get distance from them along paths
    distance = np.sqrt((x - response_x) ** 2 + (y - response_y) ** 2)
    return list(distance)


def velocity(x, y, t):
    """Returns array of velocity at each time step
    :param x: (array) x-coordinates
    :param y: (array) y-coordinates
    :param t: (array) timesteps
    :return: (list) velocities
    """
    distances = euc_dist(x, y)
    tdiff = np.ediff1d(t)
    vel = list(np.divide(distances, tdiff))
    vel.insert(0, 0)
    return vel


def maxvel_latency(vel, t):
    idx = vel.index(max(vel))
    return t[idx]


def acceleration(vel, t):
    veldiff = np.ediff1d(vel)
    tdiff = np.ediff1d(t)
    acc = list(np.divide(veldiff, tdiff))
    acc.insert(0, 0)
    return acc


def maxacc_latency(acc, t):
    idx = acc.index(max(acc))
    return t[idx]


def get_trajang(xpath, ypath, axis):
    """
    Calculate angles at each step of the trajectory with respect to x or y axis.
    :param xpath:
    :param ypath:
    :param axis:
    :return:
    """
    # TODO: replace the angles that correspond to 0 coordinates with more informative angles
    trajang = np.empty(1)
    if axis == 'x':
        trajang = np.degrees(np.arctan2(ypath, xpath))
    elif axis == 'y':
        trajang = np.degrees(np.arctan2(xpath, ypath))
    return list(trajang)


def maxang_latency(ang, t):
    idx = ang.index(max(ang))
    return t[idx]


def get_initang(x, y, ang):
    # TODO: check if this gives correct angle or if it should be the following one
    path = np.array(list(zip(x, y)))
    movement = np.all(path != 0, axis=1)
    init_ang = next(ang for (ang, movement) in zip(ang, movement) if movement is True)
    return init_ang


def get_stimang(y, ang, threshold):
    crossed = y < threshold
    stim_ang = np.array(ang)[crossed][0]
    return stim_ang


def angular_deviation(ang):
    """
    Calculate how far in degrees the trajectory deviated from a straight line between start and response at every step.
    :param ang: (array) tangent trajectory angles
    :return: (array) trajectory angles wrt to the straight line
    """
    # TODO: this could be counted wrt the actual response instead
    # response has an angle of 45 degrees, alternative response has angle of 135 degrees
    angle_to_response = abs(ang - 45)
    # angle_to_alt = abs(ang-135)
    # # TODO: normalize so that straight towards the response returns 0, and straight towards the alternative returns 1?
    # # deviation_angle = (deviation_angle - angle_to_response) / (angle_to_alt - angle_to_response)
    return list(angle_to_response)


def get_flips(coord):
    """
    Calculate the number of reversals of direction along the specified axis.
    :param coord: x- or y-coordinates
    :return: number of flips
    """
    dx = coord[:len(coord)-1] - coord[1:len(coord)]
    dx = dx/abs(dx)
    dx = dx[~np.isnan(dx)]

    dx2 = np.ediff1d(dx)
    num_flips = sum(dx2 != 0)
    return num_flips


def angle_between_points(p1, p2):
    """
    Compute angle between a line formed by two points and the origin line in radians.
    :param p1:
    :param p2:
    :return:
    """
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    return np.arctan2(y_diff, x_diff)


def rotate_point(point, origin, radians, direction):
    """
    Rotate a given point by a specified origin in a given direction.
    :param point: (array) point x- and y-coordinates to be rotated
    :param origin: (array) x- and y-coordinates of the origin
    :param radians: (int) angle of rotation
    :param direction: (str) direction of rotation, clockwise or counterclockwise
    :return: (array) new point coordinates
    """
    displacement = point - origin
    new_point = np.empty(1)
    if direction == 'clockwise':
        new_point = np.array([displacement[0] * np.cos(radians) + displacement[1] * np.sin(radians),
                              -displacement[0] * np.sin(radians) + displacement[1] * np.cos(radians)])
    elif direction == 'counterclockwise':
        new_point = np.array([displacement[0] * np.cos(radians) - displacement[1] * np.sin(radians),
                              displacement[0] * np.sin(radians) + displacement[1] * np.cos(radians)])
    rotated = new_point + origin
    return rotated


def get_max_deviation(x, y):
    """
    Return the maximum deviation away from a straight line over the course of a path, i.e. between start and end points.
    The direction of relevant deviation is always to the left of the straight line, which is towards the irrelevant
    response box.
    :param x: x-coordinates of the path
    :param y: y-coordinates of the path
    :return: distance in units of x between observed and straight line at the point of maximum deviation
    """
    path = np.array(list(zip(x, y)))
    # turn the path towards axis that goes from the start of the path vertically up
    start_point = path[0]
    end_point = path[-1]
    radians_to_rotate = math.radians(90) - angle_between_points(start_point, end_point)
    new_path = np.array(map(lambda x: rotate_point(x, path[0], radians_to_rotate, 'counterclockwise'), path))
    x_deviation = new_path[:, 0] - start_point[0]
    md = abs(min(x_deviation))
    return md


def postrapz(y, x=None, dx=1.0):
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        d = np.diff(x)
    ret = (d * (y[1:] +y[:-1]) / 2.0)
    return ret[ret>0].sum()  # The important line


def get_auc(x, y):
    path = np.array(list(zip(x, y)))
    # rotate the path towards x-axis
    start_point = path[0]
    end_point = path[-1]
    radians_to_rotate = angle_between_points(start_point, end_point)
    new_path = np.array(map(lambda x: rotate_point(x, path[0], radians_to_rotate, 'clockwise'), path))
    # use postrapz to sum the positive area under the curve only, ignore the negative area
    auc = postrapz(new_path[:,1], new_path[:,0])
    return auc


def windowmaker(xs, m):
    """
    Return all possible summed windows of sequential x-shift values.
    :param xs: (array) x-shifts vector of length N-1
    :param m: window size
    :return:
    """
    w = [xs[i:i+m] for i in range(len(xs) - m - 1)]
    return w


def pairindices(w, r):
    """
    Return list of indices of similar pairs of windows in windows list.
    :param w: (array) window lists
    :param r: tolerance value
    :return:
    """
    pi = []
    for i, s in enumerate(w[:-1]):
        for j, s2 in enumerate(w[i+1:]):
            if max(abs(np.array(s) - np.array(s2))) <= r:
                pi.append([i,j])
    return pi


def Mcounter(pim, pim1):
    """
    Return count of similar windows retained between two window sizes (e.g. m, m+1).
    :param pim: list of indices of similar windows in the smaller window size
    :param pim1: list of indices of similar windows in the larger window size
    :return:
    """
    Mm1 = 0
    for i in pim1:
        if i in pim and i in pim1:
            Mm1 += 1
    return Mm1


"""
sample entropy (Richman and Moorman, 2000); we are interested in spatial disorder along the decision axis as complexity
upon this axis should relate to competition; can be performed with PhysioNet MATLAB package; Dale et al., 2007;
choose  the window size (m; length of sequences to be compared for similarity), evidence indicates windows between
m of 3 and 6 most sensitive;
See the Supplementary Material for a Python script to calculate sample entropy and further detail.
"""


def sample_entropy(x, m, r_weight):
    """
    Return sample entropy (pairwise) of vector x.
    :param x: (array) time-normalized x-coordinates
    :param m: window size
    :param r: tolerance
    :return: sample entropy

    Sample call to calculate sample entropy for a single trajectory x, with window size m of 3,
    and a weight of tolerance r=.2.
    Recommended tolerance r is the standard deviation of x-shifts (delta_x) across conditions, we can scale it by
    some term r_weight.
    sample_entropy(x, 3, .2)

    From Supplementary Materials to Hehman, E., Stolier, R.M., and Freeman, J.B. (2014). Advanced mouse-tracking
    analytic techniques for enhancing psychological science.
    """
    # Get vector of N-1 step-wise changes in x value (xt+1 - xt)
    dx = np.ediff1d(x)
    r = r_weight * np.std(dx)
    wm = windowmaker(dx, m)[:-1]
    wm1 = windowmaker(dx, m+1)
    pim = pairindices(wm, r)
    pim1 = pairindices(wm1, r)
    Mm = len(pim)
    Mm1 = Mcounter(pim, pim1)
    e = -math.log(float(Mm1)/float(Mm))
    return e


# def sample_entropy(ts, edim=2, tau=1):
#     """
#     Calculate sample entropy of a series.
#     :param ts: a time series
#     :param edim: the embedding dimension, as for chaotic time series; a preferred value is 2
#     :param r: filter factor, work on heart rate variability has suggested setting r to be 0.2 * the data SD
#     :param elag: embedding lag, defaults to 1, more appropriately it should be set to the smallest lag at which
#     the autocorrelation function of the time series is close to zero
#     :param tau: delay time for subsampling, similar to elag
#     :return:
#
#     Ported from R function pracma:sample_entropy.
#     """
#     r = .2 * np.std(ts, ddof=1)  # ddof defaults to `0`, which gives different results than R
#     N = len(ts)
#     # edim is the window size, so we're going to make a matrix of
#     # contiguous values at the larger (edim+1) window size.
#     correl = []
#     datamat = np.zeros((edim + 1, N - edim))  # 3x98
#     for i in range(1, (edim + 1) + 1):  # 1 to the larger window size
#         datamat[i - 1] = ts[i - 1:N - edim + i - 1]  # ts[i-1:N-edim+i+1]
#     for m in [edim, edim + 1]:
#         # For window size edim, and edim+1
#         count = np.zeros((1, N - edim))
#         tempmat = datamat[:m, ]  # Windows of current size
#         for i in range(1, N - m):  # For every window...
#             a = tempmat[..., i:N - edim]
#             b = np.transpose([tempmat[..., i - 1]] * (N - edim - i))
#             X = np.abs(a - b)
#             dst = np.max(X, axis=0)
#             d = dst < r
#             count[..., i] = float(sum(d)) / (N - edim)
#         correl.append(np.sum(count) / (N - edim))
#     return np.log(correl[0] / correl[1])


""" Plotting """


def plot_trajectory(x, y, idx, axes):
    """
    Plot a single trajectory from one participant from one trial.
    :param xcoord: (array) x-coordinates to plot
    :param ycoord: (array) y-coordinates to plot
    :param idx: (int) which trial
    :param axes: (list) x- and y-limits to use
    :return:
    """
    x_path = x.iloc[idx]
    y_path = y.iloc[idx]
    plt.plot(x_path, y_path, 'b')
    plt.xlim(axes[0], axes[1])
    plt.ylim(axes[2], axes[3])


def plot_avg_traj(x, y, cond_a, cond_b, ttl):
    """
    Plot average trajectories by condition.
    :param x: (series) x-coordinates to plot
    :param y: (series) y-coordinates to plot
    :param cond_a: (str) name of condition a
    :param cond_b: (str) name of condition b
    """
    plt.plot(x[cond_a], y[cond_a], color='r', label=cond_a)
    plt.plot(x[cond_b], y[cond_b], color='b', label=cond_b)
    plt.legend()
    plt.title(ttl)


def plot_stats(meansdf, errorsdf, stat):
    """
    Plot obtained statistics.
    :param meansdf: data frame that contains calculated means
    :param errorsdf: data frame that contains calculated standard deviations
    :param stat: (str) which statistic to plot
    :return:
    """
    meansdf[stat].plot(yerr=errorsdf[stat], kind='bar')


# def plot_all(dataset, groupby, condition_a, condition_b, x='x', y='y', legend=True, title=None):
#
#
#     """Depreciated: Convenience function plotting every trajectory in 2 conditions
#
#     Parameters
#     ----------
#     dataset: Pandas DataFrame
#     groupby: string
#     The column in which the groups are defined
#     condition_a, condition_b: string
#     The labels of each group (in column groupby)
#     x, y: string, optional
#     The column names of the coordinates to be compared.
#     Default 'x', 'y'
#     legend: bool, optional
#     Include legend on plot
#     title: string, optional
#
#     Depreciated: Use:
#     ``color_map = {'condition_a': 'b', condition_b': 'r'}
#     DataFrame.apply(lambda trial: plt.plot(trial['x'], trial['y'], color_map[trial['conditon']])``
#
#     Takes a Pandas DataFrame, divides it by the grouping variable 'groupby'
#     (a string), and plots all paths in 'condition_a' in blue,
#     and 'condition_b' in red.
#     Includes a legend by default, and a title if given."""
#     for i in range(len(dataset)):
#         y_path = dataset[y].iloc[i]
#     if type(x) == list:
#         x_path = x
#     elif x == 'time':
#     x_path = range(len(y_path))
#     else:
#     x_path = dataset[x].iloc[i]
#     if dataset[groupby].iloc[i] == condition_a:
#         plt.plot(x_path, y_path, 'b')
#     elif dataset[groupby].iloc[i] == condition_b:
#     plt.plot(x_path, y_path, 'r')
#


""" Statistical Inference """

def compute_cohens_d(group1, group2):
    """Given two lists of values in group1 and group2, respectively, calculate Cohen's D,
        i.e., the effect size (how large the difference between the means is in terms of by
        how many standard deviations they differ."""
    # cohens_d = (np.mean(group1) - np.mean(group2)) / (np.sqrt((np.std(group1) ** 2 + np.std(group2) ** 2) / 2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / \
               (math.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2))
    cohens_d = round(abs(cohens_d), 2)
    return cohens_d



def get_ttest(data, grouping, stat):
    s = data[grouping].astype('category')
    cond_a = s.cat.categories[0]
    cond_b = s.cat.categories[1]
    gr1 = data[data[grouping] == cond_a]
    gr2 = data[data[grouping] == cond_b]
    return ttest_ind(gr1[stat], gr2[stat])


def bimodality_coef(samp):
    """
    Checks sample for bimodality (values > .555)
    See Freeman, J.B. & Dale, R. (2013). Assessing bimodality to detect the presence of a dual cognitive process.
    """
    n = len(samp)
    g1 = stats.skew(samp)
    g2 = stats.kurtosis(samp, fisher=True)
    b = (g1 ** 2 + 1) / (g2 + 3 * ((n - 1) ** 2 / ((n - 2) * (n - 3))))
    return b


#  this_x.append(x.iloc[p][i])
def chisquare_boolean(array1, array2):
    """Untested convenience function for chi-square test

    Parameters
    ----------
    array1, array2 : array-like
    Containing boolean values to be tested

    Returns
    --------
    chisq : float
    Chi-square value testing null hypothesis that there is an
    equal proporion of True and False values in each array.
    p : float
    Associated p-value
    """
    observed_values = np.array([sum(array1), sum(array2)])
    total_len = np.array([len(array1), len(array2)])
    expected_ratio = sum(observed_values) / sum(total_len)
    expected_values = total_len * expected_ratio
    chisq, p = stats.chisquare(observed_values, f_exp=expected_values)
    return chisq, p


def compare_means_1d(dataset, groupby, condition_a, condition_b, y='x', test='t', length=101):


    """Possibly depreciated: Compares average coordinates from two conditions using a series of t or Mann-Whitney tests.

    Parameters
    ----------
    dataset: Pandas DataFrame
    groupby: string
    The column in which the groups are defined
    condition_a, condition_b: string
    The labels of each group (in column groupby)
    y: string, optional
    The column name of the coordinates to be compared.
    Default 'x'
    test: string, optional
    Statistical test to use.
    Default: 't' (independant samples t test)
    Alternate: 'u' (Non-parametric Mann-Whitney test)

    Returns
    -----------
    t101 : t (or U) values for each point in the trajectory
    p101 : Associated p values"""
    a_x, a_y = average_path(dataset[y][dataset[groupby] == condition_a], dataset[y][dataset[groupby] == condition_a],
                            full_output=True)
    b_x, b_y = average_path(dataset[y][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b],
                            full_output=True)
    t101, p101 = [], []
    for i in range(length):
        if test == 't':  # t-test
            t, p = stats.ttest_ind(a_y[i], b_y[i])
        elif test == 'u':  # Mann-Whitney
            t, p = stats.mannwhitneyu(a_y[i], b_y[i])
        t101.append(t)
        p101.append(p)
    return t101, p101












# def uniform_time(coordinates, timepoints, desired_interval=20, max_duration=3000):
#     """Extend coordinates to desired duration by repeating the final value
#
#     Parameters
#     ----------
#     coordinates : array-like
#         1D x or y coordinates to extend
#     timepoitns : array-like
#         timestamps corresponding to each coordinate
#     desired_interval : int, optional
#         frequency of timepoints in output, in ms
#         Default 10
#     max_duration : int, optional
#         Length to extend to.
#         Note: Currently crashes if max(timepoints) > max_duration
#         Default 3000
#
#     Returns
#     ---------
#     uniform_time_coordinates : coordinates extended up to max_duration"""
#     # Interpolte to desired_interval
#     regular_timepoints = np.arange(0, timepoints[-1] + .1, desired_interval)
#     regular_coordinates = interp(regular_timepoints, timepoints, coordinates)
#     # How long should this be so that all trials are the same duration?
#     required_length = int(max_duration / desired_interval)
#     # Generate enough of the last value to make up the difference
#     extra_values = np.array([regular_coordinates[-1]] * (required_length - len(regular_coordinates) + 1))
#
#     extended_coordinates = np.concatenate([regular_coordinates, extra_values])
#     extended_timepoints = np.arange(0, max_duration + .1, desired_interval)
#     # print len(extended_coordinates), len(extended_timepoints)
#     # Return as a time series
#     return pd.TimeSeries(extended_coordinates, extended_timepoints)
