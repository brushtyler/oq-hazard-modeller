# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

# Copyright (c) 2010-2011, GEM Foundation.
#
# MToolkit is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License version 3
# only, as published by the Free Software Foundation.
#
# MToolkit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License version 3 for more details
# (a copy is included in the LICENSE file that accompanied this code).
#
# You should have received a copy of the GNU Lesser General Public License
# version 3 along with MToolkit. If not, see
# <http://www.gnu.org/licenses/lgpl-3.0.txt> for a copy of the LGPLv3 License.

"""
A set of untility functions for performing
calculations on features in an eq catalogue
"""

import numpy as np


def decimal_year(year, month, day):
    """Function to calculate the decimal year for a vector of dates"""
    marker = np.array([0., 31., 59., 90., 120., 151., 181.,
                                 212., 243., 273., 304., 334.])
    tmonth = (month - 1).astype(int)
    day_count = marker[tmonth] + day - 1.
    dec_year = year + (day_count / 365.)
    return dec_year


def greg2julian(year, month, day, hour, minute, second):
    """ Function to convert a date from Gregorian to Julian format"""
    timeut = hour + (minute / 60.0) + (second / 3600.0)
    jd = (367.0 * year) - np.floor(7.0 * (year +
             np.floor((month + 9.0) / 12.0)) / 4.0) - np.floor(3.0 *
             (np.floor((year + (month - 9.0) / 7.0) / 100.0) + 1.0) /
             4.0) + np.floor((275.0 * month) / 9.0) + day +\
             1721028.5 + (timeut / 24.0)
    return jd


def haversine(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    '''Quick function to perform geographical distance calculation
    using the haversine formula. The distance is returned in km'''
    if radians == False:
        cfact = np.pi / 180.
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2
    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) *
             (np.sin(dlon / 2.) ** 2.))
        distance[:, i] = (2. * earth_rad * np.arctan2(np.sqrt(aval),
                                                    np.sqrt(1 - aval))).T
        i += 1
    return distance
