from __future__ import division
import os
import glob
import numpy as np  # Numeric calculation
import pandas as pd  # General purpose data analysis library
import mouseanalyzer  # For mousea data
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
plt.style.use('ggplot')


"""
The input csv files have the following variables:
cuePos - cue position, 1=left, 2=right
cueColor - cue color, 1=blue, 2=red
selectedBox - which response box has been selected, 1=left, 2=right, -1=none
trial - trial number, 1 to 640
subid - subject identifier
times_ 1, times_ 2 etc. - trajectory sampling times
x_ 1, x_ 2 etc. - trajectory x coordinates
y_ 1, y_ 2 etc. - trajectory y coordinates
"""

this_dir = os.path.abspath('.')
print "Running in %s\n\
Checking for .csv files in %s" % (this_dir, os.path.join(this_dir, 'data'))

datafiles = glob.glob('data/*.csv')
print "%i files found:" % len(datafiles)
print '\n'.join(datafiles)

width = 1920
height = 1080
threshold1 = 988 # y-threshold of the upper start button boundary
threshold2 = 820  # y-threshold when the stimulus appears

# read all csv files and concatenate into a single data frame
# stack one subject data under another
data_raw = pd.concat(pd.read_csv(datafile, index_col=False) for datafile in datafiles)  # type: pd.DataFrame

nrows_raw = data_raw.shape[0]
data_raw.index = range(nrows_raw)

print "N trials in raw data: ", nrows_raw


# # convert strings to arrays
# data['t'] = map(mouseanalyzer.list_from_string, data['tTrajectory'])
# data['x'] = map(mouseanalyzer.list_from_string, data['xTrajectory'])
# data['y'] = map(mouseanalyzer.list_from_string, data['yTrajectory'])


"""
Steps to perform on raw data:
1. Space rescaling
First, all trajectories are rescaled into a standard MouseTracker coordinate space.
The top-left corner of the screen corresponds to [-1, 1.5] and the bottom-right corner corresponds to [1, 0].
In standard 2-choice designs, this leaves the start location of the mouse (the bottom-center) with coordinates [0, 0].
This standard space thus represents a 2 x 1.5 rectangle, which retains the aspect ratio of most computer screens.
"""

# select columns with time
# tcols = data.filter(regex=("times.*"))
# xcols = data.filter(regex=("^x.*"))
ycols = data_raw.filter(regex=("^y.*"))


data_raw['ix'] = ycols.apply(lambda trial: mouseanalyzer.get_good_idx(trial, threshold1, threshold2), axis=1)
data_raw['missing'] = data_raw.apply(lambda trial: mouseanalyzer.detect_missing(trial['ix'], trial['selectedBox']),
                                     axis=1)
# remove missing
data_raw = data_raw[data_raw['missing'] == False].copy()

nrows = data_raw.shape[0]

print "N trials with missing data removed: ", nrows
n_removed = nrows_raw - nrows
p_removed = n_removed/nrows_raw * 100
print "Percentage of data points removed: ", p_removed

# check if there's any particular subject to remove
print data_raw['subid'].value_counts()

# subject 5 has only 16 correct trials so needs to be removed
data_raw = data_raw[data_raw.subid != 5].copy()

# extract correct portion of trajectories
cx = data_raw.apply(lambda trial: mouseanalyzer.get_good_coord(trial.filter(regex=("^x.*")), trial['ix']), axis=1)
cy = data_raw.apply(lambda trial: mouseanalyzer.get_good_coord(trial.filter(regex=("^y.*")), trial['ix']), axis=1)
ct = data_raw.apply(lambda trial: mouseanalyzer.get_good_coord(trial.filter(regex=("times.*")), trial['ix']), axis=1)

data = data_raw[['cueColor', 'cuePos', 'selectedBox', 'subid', 'trial']].copy()
nrows = data.shape[0]
data.index = range(nrows)


data['x'] = map(lambda row: np.array(row), cx)
data['y'] = map(lambda row: np.array(row), cy)
data['t'] = map(lambda row: np.array(row), ct)

# rescale space to range x = (-1, 1) and range y = (0,1.5)
data['xscaled'] = map(lambda coordinates: mouseanalyzer.rescale_space(coordinates, -1, 1, width), data['x'])
data['yscaled'] = map(lambda coordinates: mouseanalyzer.rescale_space(coordinates, 0, 1.5, height), data['y'])


# xscaled = xcols.apply(lambda row: mouseanalyzer.rescale_space(row, -1, 1, width), axis=1)
# yscaled = ycols.apply(lambda row: mouseanalyzer.rescale_space(row, 0, 1.5, width), axis=1)
# if coordinates are in multiple columns and rescale_space returns a list:
# data['xscaled'] = data.apply(lambda row: mouseanalyzer.rescale_space(row.filter(regex=("^x.*")), -1, 1, width), axis=1)


"""
Next, for MouseTracker to average and compare trajectories, trajectories may optionally be re-mapped and overlaid.
This can be done by horizontally or vertically flipping trajectories or by rotating them.
"""
# normalize space and flip x coordinates of left trajectories to the right
data['xnorm'] = map(lambda coordinates: mouseanalyzer.normalize_x(coordinates), data['xscaled'])
data['ynorm'] = map(lambda coordinates: mouseanalyzer.normalize_y(coordinates), data['yscaled'])

# xnorm = xscaled.apply(lambda row: mouseanalyzer.normalize_x(row), axis=1)
# ynorm = yscaled.apply(lambda row: mouseanalyzer.normalize_y(row), axis=1)


"""
2. Time normalization
You can analyze mouse-tracking data with time normalization and also without (retaining in raw time). By default,
it's best to use time normalization. Time normalization is conducted because each recorded trajectory tends to have a
different length. For instance, a trial lasting 800 ms will contain approximately 56 x, y coordinate pairs, but a trial
lasting 1600 ms will contain approximately 112 x, y coordinate pairs (at 70 Hz).
To permit averaging and comparison across multiple trials with different numbers of coordinate pairs,
the x, y coordinates of each trajectory may be time-normalized into a given number of time-steps using linear
interpolation.
By default you'll time-normalize to 101 time steps to permit 100 equal time intervals. Thus, the 56 coordinate pairs
from the 800 ms trial would be fit to 101 pairs, just as the 112 pairs from the 1200 ms trial would be fit to 101 pairs.
Thus, each trajectory is normalized to have 101 time-steps and each time-step has a corresponding x and y coordinate.
"""
# data['nx'], data['ny'] = zip(*[mouseanalyzer.normalize_time(x, y, t)
#                                for x, y, t, in zip(data.xnorm, data.ynorm, data.t)])
# normalize time
nx = data.apply(lambda trial: mouseanalyzer.normalize_time(trial['xnorm'], trial['t']), axis=1)
ny = data.apply(lambda trial: mouseanalyzer.normalize_time(trial['ynorm'], trial['t']), axis=1)

# xytcols = pd.concat([xnorm, ynorm, tcols], axis=1)  # type: pd.DataFrame
# # normalize_time returns a list:
# nx = xytcols.apply(lambda trial: mouseanalyzer.normalize_time(trial.filter(regex=("^x.*")),
#                                                               trial.filter(regex=("times.*"))), axis=1)
# ny = xytcols.apply(lambda trial: mouseanalyzer.normalize_time(trial.filter(regex=("^y.*")),
#                                                               trial.filter(regex=("times.*"))), axis=1)

data['nx'] = map(lambda row: np.array(row), nx)
data['ny'] = map(lambda row: np.array(row), ny)

data['condition'] = data.apply(lambda row: mouseanalyzer.get_condition(row['cueColor'], row['cuePos']), axis=1)

"""
You can opt to retain trajectories in raw time (without time-normalization). If a raw time analysis is conducted, you
decide how many raw time bins to create between 0 ms and some cutoff (e.g., 1500 ms). You then create a user-defined
number of raw time steps. Thus, each step (i.e., coordinate pair) of a trajectory reflects the location of the mouse
during some raw time bin (e.g., 500-600 ms). Trajectories are visualized as in a normalized time analysis, but measures
of spatial attraction/curvature and complexity (MD, AUC, x-flips) are not available. Instead, velocity, acceleration,
and angle profiles are generated for each trajectory and these profiles are averaged across all trials within a
participant, separately for Condition 1 and Condition 2, and also averaged across all participants, separately for
Condition 1 and Condition 2. For the angle profiles, the current angle of movement in degrees is calculated for every
time bin, relative to the Start button.
"""

# # real time
# max_time = 5000 # Alternatively, max_time = data.rt.max()
# data['rx'] = [mouseanalyzer.uniform_time(x, t, max_duration=5000) for x, t in zip(data.x, data.t)]
# data['ry'] = [mouseanalyzer.uniform_time(y, t, max_duration=5000) for y, t in zip(data.y, data.t)]

data.to_pickle('cleandata')
