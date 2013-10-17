from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy import interpolate, interp, stats
from math import sqrt
import matplotlib.pyplot as plt

# Normalizaton
def even_time_steps(x, y, t, length = 101):
	"""Takes lists of x and y co-ordinates, and corresponding timestamps t,
	and returns these lists interpolated into 101 even time steps"""
	nt = np.arange(min(t), max(t), (float(max(t)-min(t))/length))
	nx = interp(nt, t, x)
	ny = interp(nt, t, y)
	return nx, ny, nt

def normalize_space(array, start=0, end=1):
	"""Interpolates array of 1-d coordinates to a given start and end value.
	Useful for comparison: all trajectories can be made start at (0,0), and end at (1,1)"""
	old_delta = array[-1] - array[0] # Distance between old start and end values.
	new_delta = end - start # Distance between new ones.
	# Convert coordinates to float values
	fl_array = []
	for v in array:
		fl_array.append(float(v))
	array = np.array(fl_array)
	# Finally, interpolate. We interpolate from (start minus delta) to (end plus delta)
	# to handle cases where values go below the start, or above the end values.
	normal = interp(array, [array[0] - old_delta, array[-1] + old_delta], [start - new_delta, end + new_delta])
	return normal
	
def remap_right(array):
	"""If the coordinates go leftward (end with a lower value than they 
	started with), this function will return their inverse."""
	if array[-1] - array[0] < 0:
		array_start = array[0]
		return ((array-array_start)*-1)+array_start
	else:
		return array
	
def list_from_string(string_list):
	"""Converts string represation of list '[1,2,3]' in
	an actual pythonic list [1,2,3]"""
	try:
		first = string_list.strip('[]')
		then = first.split(',')
		for i in range(len(then)):
			then[i] = int(then[i])
		return(np.array(then))
	except:
		return None
	
	
# # # Functions to apply to a set of trajectories at a time # # #

def average_path(x, y, full_output=False, length=101):
	"""Takes Pandas data columns containing lists of coordinates (paths)
	as inputs, and returns averaged paths for each one.
	Currently only works for 1- or 2-dimensional inputs"""
	# Can this be done more efficiently with .apply()?
	mx, my = [], []
	fullx, fully = [], []
	for i in range(length):
		this_x, this_y = [], []
		for p in range(len(x)):
			this_x.append(x.iloc[p][i])
			this_y.append(y.iloc[p][i])
		if full_output:
			fullx.append(this_x)
			fully.append(this_y)
		mx.append(np.mean(this_x))
		my.append(np.mean(this_y))
	if full_output:
		return fullx, fully
	return mx, my
	

def plot_means_1d(dataset, groupby, condition_a, condition_b, y = 'x', length=101, legend=True, title=None):
	"""Plots change in x axis over time.
	Assumes that dataset contains values 'x', comprising mouse
	paths standarised into 101 time steps.
	Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	(a string), and plots the average of all paths in 'condition_a' in blue,
	and the average from 'condition_b' in red.
	Includes a legend by default, and a title if given."""
	a_x, a_y = average_path(dataset[y][dataset[groupby] == condition_a], dataset[y][dataset[groupby] == condition_a], length=length)
	b_x, b_y = average_path(dataset[y][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b], length=length)
	l1 = plt.plot(a_y, color = 'r', label = condition_a)
	l2 = plt.plot(b_y, 'b', label=condition_b)
	if legend:
		plt.legend()
	plt.title(y)
	return a_y, b_y
    
    
def compare_means_1d(dataset, groupby, condition_a, condition_b, y = 'x', test = 't'):
	"""Plots change in x axis over time.
	Assumes that dataset contains values 'x', comprising mouse
	paths standarised into 101 time steps.
	Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	(a string), and plots the average of all paths in 'condition_a' in blue,
	and the average from 'condition_b' in red.
	Includes a legend by default, and a title if given."""
	a_x, a_y = average_path(dataset[y][dataset[groupby] == condition_a], dataset[y] [dataset[groupby] == condition_a], full_output=True)
	b_x, b_y = average_path(dataset[y][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b], full_output=True)
	t101, p101 = [], []
	for i in range(101):
		if test == 't':# t-test
			t, p = stats.ttest_ind(a_y[i], b_y[i])
		elif test == 'u':# Mann-Whitney
			t, p = stats.mannwhitneyu(a_y[i], b_y[i])
		t101.append(t)
		p101.append(p)
	return t101, p101

def plot_means_2d(dataset, groupby, condition_a, condition_b, x='x', y='y', length=101, legend=True, title='Average paths'):
    	"""Plots x and y coordinates.
	Assumes that dataset contains values 'x' and 'y', comprising mouse
	paths standarised into 101 time steps.
	Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	(a string), and plots the average of all paths in 'condition_a' in blue,
	and the average from 'condition_b' in red.
	Includes a legend by default, and a title if given."""
	a_x, a_y = average_path(dataset[x][dataset[groupby] == condition_a], dataset[y][dataset[groupby] == condition_a], length=length)
	b_x, b_y = average_path(dataset[x][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b], length=length)
	l1 = plt.plot(a_x, a_y, color = 'b', label = condition_a)
	l2 = plt.plot(b_x, b_y, 'r', label=condition_b)
	if legend:
		plt.legend()
	if title:
		plt.title(title)
	return a_x, a_y, b_x, b_y

def plot_all(dataset, groupby, condition_a, condition_b, x='x', y='y', legend=True, title=None):
	"""Plots all trajectories in condition_a and _b"""
	# Don't use this, use DataFrame.apply(lambda trial: plt.plot(trial['x'], trial['y'], color_map[trial['conditon']])
	for i in range(len(dataset)):
		y_path = dataset[y].iloc[i]
		if type(x) == list:
			x_path = x
		elif x == 'time':
			x_path = range(len(y_path))
		else:
			x_path = dataset[x].iloc[i]
		if dataset[groupby].iloc[i] == condition_a:
			plt.plot(x_path, y_path, 'b')
		elif dataset[groupby].iloc[i] == condition_b:
			plt.plot(x_path, y_path, 'r')
    #return?


# # # Functions to apply to a single trajectory at a time # # #
def rel_distance(x_path, y_path, full_output = False):
	"""Takes a path's x and y co-ordinates, and returns
	a list showing relative distance from each response at
	each point along path, with values closer to 0 when close to
	response 1, and close to 1 for response 2"""
	# TODO make these reference targets flexible as input
	rx1, ry1, rx2, ry2 = -1, 0, 1, 0
    	r_d, d_1, d_2 = [], [], []
    	for i in range(len(x_path)):
			x = x_path[i]
			y = y_path[i]
			# Distance from each
			d1 = sqrt( (x-rx1)**2 + (y-ry1)**2 )
			d2 = sqrt( (x-rx2)**2 + (y-ry2)**2 )
			# Relative distance
			rd = (d1 / (d1 + d2) )
			r_d.append(rd)
			if full_output:
				d_1.append(d1)
				d_2.append(d2)
	if full_output:
		return r_d, d_1, d_2
	else:
		return(r_d)

def avg_incr(series):
    d = []
    for i in range(len(series)-1):
        d.append(series[i+1] - series[i])
    return float(sum(d)) / len(d)
    
def extend_raw_path(path, target_duration=3000, t=None, rate=10):
    if type(t) == list:
        smart_t = avg_incr(t)
    path = list(path)
    for i in range( int((target_duration / rate) - len(path)) ):
        path.append(path[-1])
    return np.array(path)  

def get_init_time(y, y_limit, ascending = True):
	"""Returns the time taken for the path to go above
	y_limit (or below, if ascending is set to False)"""
	j = 0
	this_y = y[j]
	if ascending:
		while this_y < y_limit:
			# Loop until y is above the limit
			j += 1
			this_y = y[j]
	else:
		while this_y > y_limit:
			# Loop until y is above the lim
			j += 1
			this_y = y[j]
	# Return time corresponding to this y
	return(t[j])

def max_dev(x,y):
	global n, p
	# # This is treating positive and negative deviations as the same.
	# # Change this!
	startx, starty, endx, endy = 0,-1,-1,0
	ideal_x = np.arange(startx, endx, (endx-startx)*.1)
	ideal_y = np.arange(starty, endy, (endy-starty)*.1)
	ideal_x = np.append(ideal_x, endx)
	ideal_y = np.append(ideal_y, endy)
	deviations, md_signs, dev_locations = [], [], []
	for i in range(len(x)):
		distances = []
		dist_locations = []
		signs = []
		this_x = x[i]
		this_y = y[i]
		for j in range(11):
			refx = ideal_x[j]
			refy = ideal_y[j]
			dist = sqrt( (refx-this_x)**2 + (refy-this_y)**2)
			distances.append(dist)
			signs.append(np.sign(this_x-refx))
			dist_locations.append([refx, this_x])
			#print dist, dist_locations[distances.index(min(distances))
		#print len(distances), len(dist_locations)
		deviations.append(min(distances))
		md_signs.append(signs[distances.index(min(distances))])
		min_dist_loc = dist_locations[distances.index(min(distances))]
		dev_locations.append(min_dist_loc)
	md = max(deviations)
	md_sign = md_signs[deviations.index(max(deviations))]
	#md_location = dev_locations[deviations.index(md)]
	return md*md_sign #, md_location

def auc(x, y):
	areas = []
	j = len(x) - 1
	for i in range(len(x)):
		x1y2 = y[i]*x[j]
		x2y1 = x[i] * y[j]
		area = x1y2 - x2y1
		areas.append(area)
		j = i
	return float(sum(areas))/2
	
def auc2(x, y):
	areas = []
	x = list(x)
	y = list(y)
	x.append(x[-1])
	y.append(y[0])
	j = len(x) - 1
	for i in range(len(x)):
		x1y2 = y[i]*x[j]
		x2y1 = x[i] * y[j]
		area = x2y1 - x1y2 
		areas.append(area)
		j = i
	triangle = .5 * abs(x[-1] - x[0]) * abs(y[-1]*y[0])
	return float(sum(areas)) - triangle

def pythag(o, a):
	return np.sqrt( o**2 + a**2)

def velocity(x, y):
	vx = ediff1d(x)
	vy = ediff1d(y)
	vel = np.sqrt( vx**2 + vy **2 ) # Pythagoras
	return vel
    
def bimodality_coef(samp):
	n = len(samp)
	m3 = stats.skew(samp)
	m4 = stats.kurtosis(samp, fisher=True)
	#b = ( g**2 + 1) / ( k + ( (3 * (n-1)**2 ) / ( (n-2)*(n-3) ) ) )
	b=(m3**2+1) / (m4 + 3 * ( (n-1)**2 / ((n-2)*(n-3)) ))
	return b


# Make a GIF 
#~ for i in range(301):
    #~ plt.clf()
    #~ for j in range(len(data)):
        #~ if data.code.iloc[j] == 'lure':
            #~ style = 'r.'
        #~ elif data.code.iloc[j] == 'control':
            #~ style = 'b.'
        #~ else:
            #~ style = None
        #~ if style:
            #~ x = data.fx.iloc[j]
            #~ y = data.fy.iloc[j]
            #~ if len(x) > i:
                #~ plt.plot(x[i], y[i], style)
            #~ else:
                #~ plt.plot(x[-1], y[-1], style)
    #~ plt.xlim((-1.2, 1.2))
    #~ plt.ylim((-.2, 1.2))
    #~ plt.title('%ims' % (i*10))
    #~ plt.savefig(os.path.join(path, 'Step_%i.png' % (1000+i)))
# Then, using imagemagick
# convert C:\Users\40027000\Desktop\Software\ImageMagick\ImageMagick-6.8.7-0\convert -delay 10 -loop 1 *.png Output.gif
