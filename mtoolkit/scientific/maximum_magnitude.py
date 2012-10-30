# -*- coding: utf-8 -*-

# Copyright (c) 2010-2012, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.


"""
Module to implement statistically interpreted maximum magnitude
estimation algorithms.
"""

import numpy as np
from scipy.stats.mstats import mquantiles


def maximum_magnitude_analysis(year_col, magnitude_col,
    sigma_mw, maxim_mag_algorithm, iteration_tolerance,
    maximum_iterations, neq, number_samples, number_bootstraps):

    max_mag = 0
    max_mag_sigma = 0

    if maxim_mag_algorithm == 'Kijko_Npg':
        max_mag, max_mag_sigma = kijko_nonparametric_gauss(
            magnitude_col, sigma_mw, neq,
            number_samples, iteration_tolerance,
            maximum_iterations, max_observed=False)

    elif maxim_mag_algorithm == 'Cumulative_Moment':
        max_mag, max_mag_sigma = cum_mo_uncertainty(
            year_col, magnitude_col,
            sigma_mw, number_bootstraps)

    return max_mag, max_mag_sigma


def h_smooth(mag):
    """
    Function to calculate smoothing coefficient (h)
    for Gaussian Kernel estimation - based on Silverman (1986) formula.

    :param mag: Magnitude vector
    :type mag: numpy.ndarray
    :return hfact: Smoothing coefficient (h)
    :rtype hfact: Float
    """

    neq = np.float(np.shape(mag)[0])

    # Calculate inter-quartile range
    qtiles = mquantiles(mag, prob=[0.25, 0.75])
    iqr = qtiles[1] - qtiles[0]
    hfact = 0.9 * np.min([np.std, iqr / 1.34]) * (neq ** (-1. / 5.))
    # Round h to 2 dp
    hfact = np.round(100. * hfact) / 100.
    return hfact


def gauss_cdf_hastings(xval, barx=0.0, sigx=1.0):
    """
    Function to implement Hasting's approximation
    of the normalised cumulative normal function.

    :param xval: x variate
    :type xval: Float or numpy.ndarray
    :param barx: Mean of the distribution
    :type barx: Float
    :param sigx: Standard Deviation
    :type sigx: Float
    :return yval: Gaussian Cumulative Distribution
    :rtype yval: Float
    """

    x_norm = (xval - barx) / sigx

    # Fixed distribution co-efficients
    a_1 = 0.196854
    a_2 = -0.115194
    a_3 = 0.000344
    a_4 = 0.019527
    x_a = np.abs(x_norm)
    yval = 1.0 - 0.5 * (1. + a_1 * x_a + (a_2 * (x_a ** 2.)) +
                        (a_3 * (x_a ** 3.)) + (a_4 * (x_a ** 4.))) ** (-4.)
    # To deal with precision errors for tail ends
    yval[x_norm < -5.] = 0.
    yval[x_norm > 5.] = 1.
    # Finally to normalise
    yval[x_norm < 0.] = 1. - yval[x_norm < 0.]
    return yval


def kijko_npg_intfunc_simps(mval, mag, hfact, neq):
    """
    Integral function for non-parametric Gaussuan assuming that
    Simpson's rule has been invoked for exponentially spaced samples.

    :param mval: Target Magnitude
    :type mval: Float
    :param mag: Observed Magnitude values
    :type mag: numpy.ndarray
    :param hfact: Smoothing coefficient (output of h_smooth)
    :type hfact: Float
    :param neq: Number of earthquakes
    :type neq: Float
    :return intfunc: Integral of non-Parametric Gaussian function
    :rtype intfunc: Float
    """

    nmval = np.shape(mval)[0]
    mmin = np.min(mval)
    mmax = np.max(mval)

    cdf_func = np.zeros(nmval)
    for ival, target_mag in enumerate(mval):
        # Calculate normalised magnitudes
        p_min = gauss_cdf_hastings((mmin - mag) / hfact)
        p_max = gauss_cdf_hastings((mmax - mag) / hfact)
        p_mag = gauss_cdf_hastings((target_mag - mag) / hfact)
        cdf_func[ival] = ((np.sum(p_mag - p_min)) /
                          (np.sum(p_max - p_min))) ** neq
    # Now to perform integration via mid-point rule
    intfunc = 0.5 * cdf_func[0] * (mval[1] - mval[0])
    i = 1
    while i < nmval - 1:
        intfunc = intfunc + (0.5 * cdf_func[i] * (mval[i + 1] - mval[i - 1]))
        i += 1
    intfunc = intfunc + (0.5 * cdf_func[-1] * (mval[-1] - mval[-2]))

    return intfunc


def exp_spaced_points(lower, upper, nsamp):
    """
    Function to generate nsamp points
    exponentially spaced between a lower and upper bound.

    :param lower: Lower bound
    :type lower: float
    :param upper: Upper bound
    :type upper: float
    :param nsamp: Number of samples (as float)
    :type nsamp: float
    :return points: Array of exponentiall spaced points
    :rtype points: numpy.ndarray
    """

    points = np.log(np.hstack([np.exp(lower) +
            np.arange(0., nsamp - 1., 1.) * ((np.exp(upper) - np.exp(lower)) /
            (nsamp - 1.)), np.exp(upper)]))
    return points


def kijko_nonparametric_gauss(mag, mag_sig, neq,
        number_samples, iteration_tolerance,
        maximum_iterations, max_observed):
    """
    Function to implement Kijko (2004) Nonparametric Gaussian method
    for estimation of Mmax.

    :param mag: Observed magnitudes
    :type mag: numpy.ndarray
    :param mag_sig: Uncertainties on observed magnitudes
    :type mag_sig: numpy.ndarray
    :param neq: Number of earthquakes
    :type neq: Float
    :param number_samples: Number of sampling points of integral function
    :type number_samples: Integer
    :param iteration_tolerance: Intergral tolerance
    :type iteration_tolerance: Float
    :param maximum_iterations: Maximum number of Iterations
    :type maximum_iterations: Int
    :param max_observed:
        Maximum Observed Magnitude (if not in magnitude array)
        and its corresponding uncertainty (sigma)
    :type max_observed: Tuple (float) or Boolean
    :return: **mmax** Maximum magnitude and **mmax_sig** corresponding
             uncertainty
    :rtype: Float
    """

    if not(max_observed):
        # If max_observed is False then it is maximum from magnitude list
        del(max_observed)
        obsmax = np.max(mag)
        obsmaxsig = mag_sig[np.argmax(mag)]

    else:
        obsmaxsig = max_observed[1]
        obsmax = max_observed[0]

    #neq = np.shape(mag)[0]
    # Find number_eqs largest events
    if np.shape(mag)[0] <= neq:
        # Catalogue smaller than number of required events
        neq = np.float(np.shape(mag)[0])
    else:
        # Select number_eqs largest events
        mag = np.sort(mag, kind='quicksort')
        mag = mag[-neq:]
        neq = np.float(neq)

    mmin = np.min(mag)
    # Get smoothing factor
    hfact = h_smooth(mag)
    mmax = np.copy(obsmax)
    d_t = 1.E8
    j = 0
    while d_t > iteration_tolerance:
        # Generate exponentially spaced samples
        magval = exp_spaced_points(mmin, mmax, number_samples)
        # Evaluate integral function using Simpson's method
        delta = kijko_npg_intfunc_simps(magval, mag, hfact, neq)
        tmmax = obsmax + delta
        d_t = np.abs(tmmax - mmax)
        mmax = np.copy(tmmax)
        j += 1
        if j > maximum_iterations:
            print 'Kijko-Non-Parametric Gaussian estimator reached'
            print 'maximum # of iterations'
            d_t = 0.5 * iteration_tolerance
    mmax_sig = np.sqrt(obsmaxsig ** 2. + delta ** 2.)
    # mmax[()] temporary fix, mmax is a numpy array not a float,
    # mmax[()] returns the float number inside the zero rank array.
    return mmax[()], mmax_sig


def cumulative_moment(year, mag):
    """
    Calculation of Mmax using aCumulative Moment approach, adapeted from
    the cumulative strain energy method of Makropoulos & Burton (1983).

    :param year: Year of Earthquake
    :type year: numpy.ndarray
    :param mag: Magnitude of Earthquake
    :type mag: numpy.ndarray
    :return mmax: Returns Maximum Magnitude
    :rtype mmax: Float
    """

    # Calculate seismic moment
    m_o = 10. ** (9.05 + 1.5 * mag)
    year_range = np.arange(np.min(year), np.max(year) + 1, 1)
    nyr = np.float(np.shape(year_range)[0])
    morate = np.zeros(nyr, dtype=float)
    # Get moment release per year
    for loc, tyr in enumerate(year_range):
        idx = np.abs(year - tyr) < 1E-5
        if np.sum(idx) > 0:
            # Some moment release in that year
            morate[loc] = np.sum(m_o[idx])
    ave_morate = np.sum(morate) / nyr

    # Average moment rate vector
    exp_morate = np.cumsum(ave_morate * np.ones(nyr))
    modiff = np.abs(np.max(np.cumsum(morate) - exp_morate)) + \
                    np.abs(np.min(np.cumsum(morate) - exp_morate))
    # Return back to Mw
    mmax = (2. / 3.) * (np.log10(modiff) - 9.05)
    return mmax


def cum_mo_uncertainty(year, mag, sigma_m, number_bootstraps, seed=None):
    """
    Function to calculate mmax wth uncertainty using the cumulative moment
    formulation (adapted from Makropoulos & Burton, 1983) with bootstrapping.

    :param year: Year of earthquake
    :type year: numpy.ndarray
    :param mag: Magnitude of earthquake
    :type mag: numpy.ndarray
    :param sigma_m: Uncertainties on the magnitudes
    :type sigma_m: numpy.ndarray
    :param nbootstrap: Number of samples for bootstrapping
    :type nbootstrap: Integer
    :keyword seed: fixed seed number
    :type seed: Integer
    :return: **mmax** Maximum magnitude and **mmax_sig** corresponding
             uncertainty
    :rtype: Float
    """

    if isinstance(seed, int):
        sampler = np.random.RandomState(seed)
    else:
        sampler = np.random

    neq = np.shape(mag)[0]
    mmax_samp = np.zeros(number_bootstraps)
    for i in range(0, number_bootstraps):
        mw_sample = mag + sigma_m * sampler.normal(0, 1, neq)
        mmax_samp[i] = cumulative_moment(year, mw_sample)

    # Return mean and standard deviation of sample
    return np.mean(mmax_samp), np.std(mmax_samp, ddof=1)
