from __future__ import division
import numpy as np  # Numeric calculation
import pandas as pd  # General purpose data analysis library
import mouseanalyzer  # For mousea data
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
plt.style.use('ggplot')

data = pd.read_pickle('cleandata')

"""
3. Averaging - each participant's mean trajectory for each condition is computed by averaging all x- and y-points at
each time step.
Each participant's mean trajectory for one condition is computed by averaging together all the x coordinates of
trajectories in that condition at each time-step, and all the y coordinates of trajectories in that condition at each
time-step. If doing a normalized time analysis, then averaging is very intuitive. Each participant's mean trajectory
for one condition has a user-defined number of x, y coordinate pairs. The first pair (e.g., 1) reflects the start of
mouse movement and the last pair (e.g., 101) reflects the end of mouse movement when a response was clicked on.
This would be true across all trials regardless of how long participants actually took. This is why time normalization
is valuable.
If doing a raw time analysis, then each participant's mean trajectory for one condition has a user-defined number of
x, y coordinate pairs that each correspond with a raw time bin (e.g., 500-600 ms) up until the cutoff. What this means
is mouse trajectory data that persists after the cutoff are discarded (with the trajectory data for that trial ending
in mid-flight). Thus, if the raw time cutoff is 1500 ms (and we have, for instance, 20 equal raw time bins until
1500 ms), then the section of mouse trajectories that take place after 1500 ms will be discarded. Although this makes
the analysis more difficult, a raw time analysis can be extremely useful for studies involving time-sensitive stimuli
(e.g., spoken words or compound sequences of images, etc.) or simply if it is important to know in raw time (ms) where
and what the mouse is doing.
"""


""" Make exploratory plots """

# mouseanalyzer.plot_trajectory(data['xnorm'], data['ynorm'], 1, [-1, 1, 0, 1.5])
# mouseanalyzer.plot_trajectory(data['nx'], data['ny'], 1, [-1, 1, -0.5, 1.5])

subjects = [x for x in range(1, 21) if x != 5]
for i in subjects:
    sx = data.groupby(['subid', 'condition']).apply(lambda row: np.nanmean(row['nx']))
    sy = data.groupby(['subid', 'condition']).apply(lambda row: np.nanmean(row['ny']))
    plt.plot(sx.xs((i,'congruent')), sy.xs((i,'congruent')), color='r')
    plt.plot(sx.xs((i,'incongruent')), sy.xs((i,'incongruent')), color='b')


sx = mouseanalyzer.average_path(data, 'nx', 'condition')
sy = mouseanalyzer.average_path(data, 'ny', 'condition')
mouseanalyzer.plot_avg_traj(sx, sy, 'congruent', 'incongruent', 'tt')



# sx2 = mouseanalyzer.average_path(data, 'nx', 'subid')
# sy2 = mouseanalyzer.average_path(data, 'ny', 'subid')

# tmean = data.groupby('condition').apply(lambda row: np.mean(row['nx']))
# terror = data.groupby('condition').apply(lambda row: np.std(row['nx']))
# meansdf[stat].plot(yerr=errorsdf[stat], kind='bar')


""" Trajectory Dependent Variables """

"""
maxx	        Maximum x-value observed
minx	        Minimum x-value observed
maxy	        Maximum y-value observed
miny	        Minimum y-value observed
"""

data['maxx'] = data.apply(lambda trial: max(trial['xnorm']), axis=1)
data['minx'] = data.apply(lambda trial: min(trial['xnorm']), axis=1)
data['maxy'] = data.apply(lambda trial: max(trial['ynorm']), axis=1)
data['miny'] = data.apply(lambda trial: min(trial['ynorm']), axis=1)


"""
tdur        The total duration of the trajectory
latency     The latency of the start of the movement
tmove       The total time of motion
dwell       The dwell time to commit to a final response
inittime    The time when the y-threshold was crossed and stimulus appeared
"""

# this will specify the dwell size as proportion of the possible trajectory reach on rescaled dimensions,
# i.e. 1 for x-axis and 1.5 for y-axis
# TODO: change to specifying it in terms of pixels
dwell_size = 0.1
data['tdur'] = data.apply(lambda trial: mouseanalyzer.total_duration(trial['t']), axis=1)
data['latency'] = data.apply(lambda trial: mouseanalyzer.get_latency(trial['x'], trial['y'], trial['t']), axis=1)
data['tmove'] = data.apply(lambda trial: mouseanalyzer.total_inmotion(trial['x'], trial['y'], trial['t']), axis=1)
data['dwell'] = data.apply(lambda trial: mouseanalyzer.get_dwell(trial['xnorm'], trial['ynorm'],
                                                                 trial['t'], dwell_size), axis=1)
data['inittime'] = data.apply(lambda trial: mouseanalyzer.get_init_time(trial['t'], trial['y'], threshold), axis=1)


"""
eucdist	        Euclidean distance of the trajectory
distinmot	    Distance traveled outside of the escape region
respdist        Distance from response for each step of the trajectory
foildist        Distance from foil for each step of the trajectory
"""

data['eucdist'] = data.apply(lambda trial: sum(mouseanalyzer.euc_dist(trial['x'], trial['y'])), axis=1)
# TODO: implement distinmot?
data['respdist'] = data.apply(lambda trial: mouseanalyzer.dist_from_response(trial['xnorm'], trial['ynorm']), axis=1)
data['foildist'] = data.apply(lambda trial: mouseanalyzer.dist_from_response(trial['x'], trial['y'], foil=True), axis=1)


"""
Abrupt shifts in trajectory were marked by an initial spike in velocity towards one response, followed by a second
spike in velocity after the reversal in direction.

vel             Velocity at each time step
maxvel	        The maximum velocity reached
maxvelstart	    The latency when maximum velocity was observed
acc             Acceleration at each time step
maxacc	        Maximum acceleration
maxaccstart	    The latency when maximum acceleration was observed
"""

data['vel'] = data.apply(lambda trial: mouseanalyzer.velocity(trial['x'], trial['y'], trial['t']), axis=1)
data['maxvel'] = data.apply(lambda trial: max(trial['vel']), axis=1)
data['maxvelstart'] = data.apply(lambda trial: mouseanalyzer.maxvel_latency(trial['vel'], trial['t']), axis=1)

data['acc'] = data.apply(lambda trial: mouseanalyzer.acceleration(trial['vel'], trial['t']), axis=1)
data['maxacc'] = data.apply(lambda trial: max(trial['acc']), axis=1)
data['maxaccstart'] = data.apply(lambda trial: mouseanalyzer.maxacc_latency(trial['acc'], trial['t']), axis=1)


"""
trajang	        Angles tangent to the trajectory
maxang	        Maximum severity of angle towards incorrect response button
maxangstart	    The latency when maximum severity of angle was observed
initang	        The initial trajectory angle after movement start
stimang	        The initial trajectory angle after the appearance of the stimulus
angdev          Angular deviation from the straight line that goes from the start to response
"""

data['trajang'] = data.apply(lambda trial: mouseanalyzer.get_trajang(trial['xnorm'], trial['ynorm'], 'x'), axis=1)
data['maxang'] = data.apply(lambda trial: max(trial['trajang']), axis=1)
data['maxangstart'] = data.apply(lambda trial: mouseanalyzer.maxang_latency(trial['trajang'], trial['t']), axis=1)
data['initang'] = data.apply(lambda trial: mouseanalyzer.get_initang(trial['x'], trial['y'], trial['trajang']), axis=1)
data['stimang'] = data.apply(lambda trial: mouseanalyzer.get_stimang(trial['y'], trial['trajang'], threshold), axis=1)
data['angdev'] = data.apply(lambda trial: mouseanalyzer.angular_deviation(np.array(trial['trajang'])), axis=1)


"""
4. Measuring spatial attraction to assess the degree of attraction toward an unselected response, indexing the magnitude
of activation for each response option as the decision process unfolds over time.
These preprocessed and averaged mouse trajectory data could be used in many ways and which ways are used depends on
the research questions at hand. In many cases, one question is whether the trajectories for one condition travel
reliably more closely to an unselected response relative to another condition. Or, it might be useful to know simply
how much deviation or curvature exists in the trajectory in general.
Prior studies have used two measures, maximum deviation and area-under-the-curve.
For both of these measures, we first computes an idealized response trajectory (a straight line between each
trajectory's start and endpoints). The MD of a trajectory is then calculated as the largest perpendicular deviation
between the actual trajectory and its idealized trajectory out of all time-steps. Thus, the higher the MD, the more the
trajectory deviated toward the unselected alternative. The AUC of a trajectory is calculated as the geometric area
between the actual trajectory and the idealized trajectory (straight line). Area on the opposite side (i.e., in the
direction away from the unselected response) of the straight line is calculated as negative area.

md              Maximum deviation of the trajectory towards incorrect response button
auc	            Area under the Curve
DVmaxpull	    Maximum pull towards the incorrect response button
DVmaxpullstart	Latency of pull towards incorrect response button
"""

data['md'] = data.apply(lambda trial: mouseanalyzer.get_max_deviation(trial['nx'], trial['ny']), axis=1)
data['auc'] = data.apply(lambda trial: mouseanalyzer.get_auc(trial['nx'], trial['ny']), axis=1)


"""
5. Measuring complexity (spatial disorder analysis) - usually more complexity when the response is ambiguous and
uncertain, when there's more or less equal attraction to both alternatives; complexity in response trajectories may be
taken as evidence for a formal dynamical process at work.
In some cases, it may be helpful to know how complex trajectories are. For example, if an unselected alternative
simultaneously acts as another attractor that exerts a force on participants' mouse trajectories, this additional
stress might manifest as less smooth, more complex and fluctuating trajectories.
We calculate x-flips and y-flips, which are the number of reversals of direction along the x-axis and y-axis,
respectively. This captures the fluctuations in the hand's vacillation between response alternatives along the axes.

xflips	    Change in x-direction
yflips	    Change in y-direction
xangflips   Change of angle-flipping in x-axis
xent	        Entropy along x-axis
yent	        Entropy along y-axis
aent	        Entropy along angle-trajectory
"""

# TODO: these can be calculated separately for different trajectory regions: latency, motion, dwell
data['xflips'] = data.apply(lambda trial: mouseanalyzer.get_flips(trial['x']), axis=1)
data['yflips'] = data.apply(lambda trial: mouseanalyzer.get_flips(trial['y']), axis=1)
data['xangflips'] = data.apply(lambda trial: mouseanalyzer.get_flips(np.array(trial['trajang'])), axis=1)

data['xent'] = data.apply(lambda trial: mouseanalyzer.sample_entropy(trial['nx'], 3, .2), axis=1)
data['yent'] = data.apply(lambda trial: mouseanalyzer.sample_entropy(trial['ny'], 3, .2), axis=1)
data['aent'] = data.apply(lambda trial: mouseanalyzer.sample_entropy(trial['trajang'], 3, .2), axis=1)


"""
Outlier measures
ol1     Binary value indicating whether motion time was longer than latency
ol2	    Binary value indicating whether maximum velocity was inside the latency region
ol3	    Binary value indicating whether maximum acceleration is inside the latency region
ol4	    Binary value indicating whether trajectory travels in negative y-values after escaping the latency region
"""

# Is motion time longer than latency time? Gets at whether cognitive processing is mostly occurring in latency region.
data['ol1'] = data.apply(lambda trial: trial['latency'] < trial['tmove'], axis=1)

# # Is max velocity inside latency region? Gets at whether strongest commitment to a response occurs in latency region.
# TODO: define latency region and get index
# data['ol2'] = data.apply(lambda trial: trial['maxvel'].index < latency.index, axis=1)
# # Is max acceleration inside of latency region? (Same reason as velocity)
# data['ol3'] = data.apply(lambda trial: trial['maxacc'].index < latency.index, axis=1)
# TODO: what do negative movements mean?
# # Does trajectory dive below y axis after escaping latency region? Gets at whether the trajectory is particularly wild.
# data['ol4'] = data.apply(lambda trial: trial['latency'] < trial['tmove'], axis=1)


""" Trajectory Stats """

grouped = data.groupby(['subid', 'condition'])
means = grouped.mean()
errors = grouped.std()

# mouseanalyzer.plot_stats(means, errors, 'auc')
# mouseanalyzer.get_ttest(data, 'condition', 'auc')

gr1 = means.xs('congruent', level='condition')['auc']
gr2 = means.xs('incongruent', level='condition')['auc']
auct = ttest_rel(gr1, gr2)
aucd = mouseanalyzer.compute_cohens_d(gr1, gr2)

gr1 = means.xs('congruent', level='condition')['md']
gr2 = means.xs('incongruent', level='condition')['md']
mdt = ttest_rel(gr1, gr2)
mdd = mouseanalyzer.compute_cohens_d(gr1, gr2)

# power analysis in R:
# pwr.t.test(n=20, d=1.22, sig.level=1.7726744801232607e-08, type='paired', alternative='greater')
"""
Paired t test power calculation

              n = 20
              d = 1.22
      sig.level = 0.0004255762
          power = 0.9044376
    alternative = greater

NOTE: n is number of *pairs*
"""

"""
6. Calculating the bimodality coefficient - to detect that e.g. half of the trajectories in the condition of interest
are extremely attracted and other half not at all; i.e. there can be discrete-like errors that when averaged with no
error trajectories, create an appearance of continuous attraction - calculate the bimodality coefficient b
(check if b > 0.555) and plot the AUC histograms
You may wish to examine the distribution of trajectories' trial-by-trial spatial attractions toward an unselected
alternative (indexed by MD or AUC). This can be especially useful for formally determining the temporal nature of one
condition's stronger attraction toward an unselected alternative relative to another condition. For instance, suppose
a researcher finds that trajectories in Condition 1 are continuously more attracted toward the unselected alternative
than trajectories in Condition 2. This is visually apparent by plotting the two mean trajectories and statistically
apparent by a significant difference in MD or AUC between Condition 1 and Condition 2. Underlying this reliable
continuous attraction effect, however, could be a subpopulation of discrete-like errors biasing the results.
For instance, if half the trajectories in Condition 1 headed straight to the selected alternative, and the other half
initially headed straight to the unselected alternative, followed by a sharp midflight correction redirecting the
trajectory toward the selected alternative, the mean trajectories would exhibit a reliable attraction effect that
appeared continuous although it was actually caused by several discrete-like errors. If such a subpopulation of
discrete-like errors were biasing the results, the distribution of Condition 1 would be bimodal (some trajectories
show zero attraction and the other trajectories show extreme attraction).

Bimodality may be tested by calculating the bimodality coefficient (b) and determining whether b > 0.555.
If b > 0.555, the distribution is considered to be bimodal, and if b <= 0.555, it is considered to be unimodal.
For testing bimodality, first z-normalize the MD and AUC values of trials within each participant, together
across Condition 1 and Condition 2.

You may also want to further alleviate concerns about latent bimodality by ensuring that the shapes of the two
distributions are statistically indistinguishable.
To accomplish this, the Kolmogorov-Smirnov test is used available on the web:
http://www.physics.csbsju.edu/stats/KS-test.html).
For use with this test, also z-normalize the MD and AUC values of trials within each participant, separately across
Condition 1 and Condition 2. This test, unlike the bimodality test, is inferential; if p < .05, the shapes of the two
distributions reliably depart from one another.
"""


"""
7. PCA is ideal for identifying unique components within a mousetrajectory, in which the data points within each
component may be more correlated than with data points in other components. These separate components may index unique
psychological constructs. PCA thus provides a method to identify and extract multiple components from averaged
mouse-trajectories for purposes of submitting them to further analysis.

- can use x-coordinates to perform PCA with some software
- upon running the analysis, researchers would examine the "Rotated Component Matrix" in the output.
These are the unique components identified in the mouse-trajectory data. Loadings on each of these components can then
be plotted over time for interpretation.
- component matrices can then be used to guide the creation of new variables, using the time bins that load highest on
each component.
"""

"""
8. Raw time analysis, velocity, acceleration
- velocity and acceleration calculations require raw time analysis
- split the data into time bins:
"Time bins" specifies the amount of data points reported across a single trajectory (e.g., for a 1,000 ms trajectory,
20 time bins would equal 20 data points of x- and y-coordinate information averaged across 50 ms);
we then tested whether the x-coordinate was significantly correlated with the electoral success of each candidate at
each time bin
- instead of just looking at the x-coordinate, Euclidean distance can be used: the Euclidean (i.e., straight line)
distance between the cursor and response option, incorporating both x- and y-coordinates;
better use proportional proximity: 1 - distance / max (distance);
For example, Spivey et al. (2005) examined what point in time the mouse's proximity to the selected response
significantly differed from its proximity to the unselected response.
- stronger competition between response options should be characterized by an initial decreased velocity as competing
choices inhibit each other, followed by an increase in velocity once the system converges upon a decision and the
inhibition is alleviated
- velocity and acceleration may also reflect the degree of response activation and thus allow for inferences about when
commitments to a particular response are made; velocity is calculated as the distance between subsequent coordinates at
different raw time points, and acceleration may be computed from changes in velocity across time points
"""


""" Save Output """


# data.to_csv('processed.csv', index=False)
# print "Summary statistics saved to %s" % os.path.join(this_dir, 'processed.csv')

# save stat results too


# # reverse y axis because in pixels up is lower number
# # data['y'] = height - data.y  # this is useful for plotting actual trajectories
# # normalize x and y to the origin, i.e. all trajectories start from point (0,0)
# # we also flip the leftward responses in order to perform all analyses on one side
# data['xnorm'] = map(lambda coordinates: mouseanalyzer.normalize_x(coordinates), data['x'])
# data['ynorm'] = map(lambda coordinates: mouseanalyzer.normalize_y(coordinates), data['y'])
# mouseanalyzer.plot_trajectory(data['xnorm'], data['ynorm'], 1, [-width/2, width/2, 0, 1080])
# # normalize space
# data['xscaled'] = map(lambda coordinates: mouseanalyzer.rescale_space(coordinates, -1, 1, width), data['xnorm'])
# data['yscaled'] = map(lambda coordinates: mouseanalyzer.rescale_space(coordinates, 0, 1.5, height), data['ynorm'])
# are these the same as trajectory distance and max deviation?
# arclengthtotal	The length of the arc subtending the trajectory after motion was initiated
# maxpathoff	    Maximum offset of the trajectory

# For the demo data, the input csv files have the following variables:
# accuracy - correctness of the response {1, 0}; 1 if (condition=="lie" and response=="2") or
#             (condition=="truth" and response=="1")
# condition - which condition in the trial {lie, truth}
# count_trial_sequence - trial number
# height - height of the screen
# probe - stimulus presented
# response - response given {1, 2}
# rt - reaction time
# subject_nr - subject identifier
# tTrajectory - trajectory sampling times
# width - width of the screen
# xTrajectory - trajectory x coordinates
# yTrajectory - trajectory y coordinates
