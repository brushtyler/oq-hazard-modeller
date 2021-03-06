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
The purpose of this module is to provide objects
to read csv files containing eq definitions and
create eq entries.
"""

import os
from datetime import datetime as time
from csv import DictReader, DictWriter

FIELDNAMES = ['eventID', 'Agency', 'Identifier',
              'year', 'month', 'day',
              'hour', 'minute', 'second',
              'timeError', 'longitude', 'latitude',
              'SemiMajor90', 'SemiMinor90', 'ErrorStrike',
              'depth', 'depthError', 'Mw',
              'sigmaMw', 'Ms', 'sigmaMs',
              'mb', 'sigmamb', 'ML',
              'sigmaML']


class MalformedCatalogError(Exception):
    """
    Malformed catalog error could be raised
    because of an invalid csv earthquake
    catalogue.
    """

    def __init__(self):
        header = ','.join(FIELDNAMES)
        msg = ('The fieldnames should be placed on top of the catalogue'
            "file, the valid header is '%s' without quotes." % (header))
        Exception.__init__(self, msg)


class EqEntryReader(object):
    """
    EqEntryReader allows to read and create
    eq entries dictionaries in an iterative way.
    Every read eq entry is composed by a series
    of attributes which need conversion
    (int or float) and validation through a
    series of checks. If a compulsory field
    can't be converted or doesn't pass
    it's own check, no eq entry is generated,
    and an exception is raised, otherwise if
    a non compulsory field can't be converted
    or validated an empty string is placed instead
    of the incorrect value.
    """

    EMPTY_STRING = ''

    def __init__(self, eq_entries_source):
        """
        to_int   - fields to be converted in integer
        to_float - fields to be converted in float
        check_map - associates each field with its own check
        current_line - denotes the line in use by the read method
        """
        self.validate_csv_catalog(eq_entries_source)

        self.eq_entries_source = eq_entries_source

        self.to_int = ['eventID', 'Identifier', 'year', 'month',
                'day', 'hour', 'minute']

        self.to_float = ['second', 'timeError', 'longitude',
                'latitude', 'SemiMajor90', 'SemiMinor90',
                'ErrorStrike', 'depth', 'depthError',
                'Mw', 'sigmaMw', 'Ms',
                'sigmaMs',  'mb', 'sigmamb',
                'ML', 'sigmaML']

        self.compulsory_fields = self.to_int + [self.to_float[2],
                self.to_float[3], self.to_float[7], self.to_float[9]]

        self.check_map = {
                'eventID': self.check_positive_value,
                'Agency': self.no_check,
                'Identifier': self.check_positive_value,
                'year': self.check_year,
                'month': self.check_month,
                'day': self.check_day,
                'hour': self.check_hour,
                'minute': self.check_minute,
                'second': self.check_second,
                'timeError': self.no_check,
                'longitude': self.check_longitude,
                'latitude': self.check_latitude,
                'SemiMajor90': self.check_positive_value,
                'SemiMinor90': self.check_positive_value,
                'ErrorStrike': self.check_epicentre_error_location,
                'depth': self.check_positive_value,
                'depthError': self.check_positive_value,
                'Mw': self.no_check,
                'sigmaMw': self.check_sigma_mw,
                'Ms': self.no_check,
                'sigmaMs': self.check_positive_value,
                'mb': self.no_check,
                'sigmamb': self.check_positive_value,
                'ML': self.no_check,
                'sigmaML': self.check_positive_value
        }

        self.current_line = 0

    @classmethod
    def validate_csv_catalog(cls, eq_entries_source):
        """
        Check if the earthquake catalogue
        contains a valid header, constituted
        by the denoted fieldnames.
        """
        fieldnames = [elem.strip()
                      for elem in eq_entries_source.readline().split(',')]
        if fieldnames != FIELDNAMES:
            raise MalformedCatalogError()

    def read(self):
        """
        Return a generator that provides an eq
        entry in a dictionary for every line
        with valid values.
        """
        eq_reader = DictReader(self.eq_entries_source, fieldnames=FIELDNAMES)
        # eq definitions start at line 2
        for self.current_line, eq_line in enumerate(eq_reader, start=2):
            eq_entry = self.convert_values(eq_line)
            for field in eq_entry.keys():
                if not self.check_map[field](field, eq_entry) and (field in
                    self.compulsory_fields):

                    raise EqEntryValidationError(field, eq_entry[field],
                        self.current_line)
            yield eq_entry

    def read_eq_catalog(self):
        """
        Return a list of eq entry
        representing the earthquake
        catalogue.
        """

        eq_catalog = []
        for eq_entry in self.read():
            eq_catalog.append(eq_entry)
        return eq_catalog

    def convert_values(self, dict_fields_values):
        """
        Return an eq dictionary with all fields
        values converted to the respective type
        """

        for key in dict_fields_values:
            if key in self.to_int:
                try:
                    dict_fields_values[key] = int(
                        dict_fields_values[key])
                except ValueError:
                    # fields in self.to_int are all compulsory
                    raise EqEntryValidationError(key,
                            dict_fields_values[key], self.current_line)

            elif key in self.to_float:

                try:
                    dict_fields_values[key] = float(
                        dict_fields_values[key])
                except ValueError:
                    if key in self.compulsory_fields:
                        raise EqEntryValidationError(key,
                            dict_fields_values[key])
                    else:
                        dict_fields_values[key] = EqEntryReader.EMPTY_STRING

        return dict_fields_values

    def no_check(self, field, eq_entry):
        """
        Return True without applying any check.
        """

        return True

    def check_positive_value(self, field, eq_entry):
        """
        Return a bool stating if check is passed,
        if a non compulsory field doesn't pass the check
        an empty string is placed instead of the value
        in the eq_entry.
        """

        if field in self.compulsory_fields:
            return eq_entry[field] > 0
        if eq_entry[field] < 0:
            eq_entry[field] = EqEntryReader.EMPTY_STRING
        return True

    def check_year(self, field, eq_entry):
        """
        Return a bool stating if check is passed.
        """

        return -10000 <= eq_entry[field] <= time.now().year

    def check_month(self, field, eq_entry):
        """
        Return a bool stating if check is passed.
        """

        return 1 <= eq_entry[field] <= 12

    def check_day(self, field, eq_entry):
        """
        Return a bool stating if check is passed.
        """

        return (eq_entry['month'] == 2 and eq_entry[field] <= 29)\
            or (eq_entry['month'] != 2 and 1 <= eq_entry[field] <= 31)

    def check_hour(self, field, eq_entry):
        """
        Return a bool stating if check is passed.
        """

        return 0 <= eq_entry[field] <= 23

    def check_minute(self, field, eq_entry):
        """
        Return a bool stating if check is passed.
        """

        return 0 <= eq_entry[field] <= 59

    def check_second(self, field, eq_entry):
        """
        Return a bool stating that a check is applied,
        if the non compulsory field second doesn't pass
        the check an empty string is placed instead
        of the value in the eq_entry.
        """

        if not 0 <= eq_entry[field] <= 59:
            eq_entry[field] = EqEntryReader.EMPTY_STRING
        return True

    def check_longitude(self, field, eq_entry):
        """
        Return a bool stating if check is passed.
        """

        return -180 <= eq_entry[field] <= 180

    def check_latitude(self, field, eq_entry):
        """
        Return a bool stating if check is passed.
        """

        return -90 <= eq_entry[field] <= 90

    def check_epicentre_error_location(self, field, eq_entry):
        """
        Return a bool stating that some checks are applied,
        if one of the three non compulsory fields doesn't pass
        it's own check an empty string is placed instead
        of the values in the three correlated fields
        (i.e. ErrorStrike, SemiMinor90, SemiMajor90).
        """

        if not 0 <= eq_entry[field] <= 360 or \
            eq_entry['SemiMinor90'] == EqEntryReader.EMPTY_STRING or \
            eq_entry['SemiMajor90'] == EqEntryReader.EMPTY_STRING or \
            not (eq_entry['SemiMinor90'] <= eq_entry['SemiMajor90']):
            eq_entry['SemiMinor90'] = eq_entry['SemiMajor90'] \
            = eq_entry[field] = EqEntryReader.EMPTY_STRING

        return True

    def check_sigma_mw(self, field, eq_entry):
        """
        Return a bool stating that some checks are applied
        if the non compulsory field sigmaMw doesn't pass
        the check a default value is placed instead
        of the value in the eq_entry.
        """

        if eq_entry[field] == EqEntryReader.EMPTY_STRING or \
            eq_entry[field] < 0:

            eq_entry[field] = 0.0

        return True


class EqEntryValidationError(Exception):
    """
    EqEntry validation error could be raised
    because of a failed conversion or check
    of an eq entry compulsory field.
    """

    def __init__(self, field, value, line_number):
        """Constructs a new validation exception
        for the given eq entry field"""

        msg = "Validation error with the field: %s, having value: %s "\
        "at line number: %s" % (field, value, line_number)
        Exception.__init__(self, msg)
        self.args = (field, msg)


class EqEntryWriter(object):
    """
    EqEntryWriter allows the user to write
    eq entries in a csv file
    """

    def __init__(self, output_filename):
        output_dir = os.path.dirname(output_filename)
        if not os.path.exists(output_dir):
            raise IOError('Invalid dir: %s' % output_dir)
        self.output_filename = output_filename

    def write_rows(self, entries):
        """Write rows in the csv file"""

        with open(self.output_filename, 'w') as output_file:
            writer = DictWriter(output_file, FIELDNAMES)
            writer.writeheader()
            writer.writerows(entries)
