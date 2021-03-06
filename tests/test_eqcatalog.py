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


import unittest
import filecmp
from StringIO import StringIO

from mtoolkit.eqcatalog import (EqEntryReader, EqEntryWriter,
                                MalformedCatalogError, EqEntryValidationError)

from nrml.nrml_xml import get_data_path, DATA_DIR

FIELDNAMES = ['eventID', 'Agency', 'Identifier',
              'year', 'month', 'day',
              'hour', 'minute', 'second',
              'timeError', 'longitude', 'latitude',
              'SemiMajor90', 'SemiMinor90', 'ErrorStrike',
              'depth', 'depthError', 'Mw',
              'sigmaMw', 'Ms', 'sigmaMs',
              'mb', 'sigmamb', 'ML',
              'sigmaML']


class EqEntryReaderTestCase(unittest.TestCase):

    def setUp(self):

        self.first_data_row = [1, 'AAA', 20000102034913,
                                2000, 01, 02,
                                03, 49, 13,
                                0.02, 7.282, 44.368,
                                2.43, 1.01, 298,
                                9.3, 0.5, 1.71,
                                0.355, '', '',
                                '', '', 1.7, 0.1]

        self.data_row_to_convert = ['2', 'AAA', '20000105132157',
                                    '2000', '01', '05',
                                    '13', '21', '57',
                                    '0.10', '11.988', '44.318',
                                    '0.77', '0.25', '315',
                                    '7.9', '0.5', '3.89',
                                    '0.199', '   ', '   ',
                                    '3.8', '0.1', '   ', '   ']

        self.eq_reader = EqEntryReader(open(get_data_path('ISC_small_data.csv',
                    DATA_DIR)))

    def test_invalid_csv_file_raise_exc(self):
        invalid_cat = StringIO('1,2,3,4,5,6,7,8')
        self.assertRaises(MalformedCatalogError, EqEntryReader, invalid_cat)

    def test_generated_eq_entry(self):
        first_eq_entry = dict(zip(FIELDNAMES, self.first_data_row))

        self.assertEqual(first_eq_entry,
            self.eq_reader.read().next())

    def test_an_incorrect_conversion_raise_exception(self):
        dict_bad_eventid_value = {'eventID': 'a'}
        dict_bad_year_value = {'year': '45os'}
        dict_bad_semimajor_value = {'SemiMajor90': '45as'}

        dict_good_event_id_value = {'eventID': '1234'}
        dict_good_year_value = {'year': '3000'}

        converted_id_value = self.eq_reader.convert_values(
            dict_good_event_id_value)['eventID']
        converted_year_value = self.eq_reader.convert_values(
            dict_good_year_value)['year']
        converted_semimajor_value = self.eq_reader.convert_values(
            dict_bad_semimajor_value)['SemiMajor90']

        self.assertRaises(EqEntryValidationError,
            self.eq_reader.convert_values, dict_bad_eventid_value)
        self.assertRaises(EqEntryValidationError,
            self.eq_reader.convert_values, dict_bad_year_value)

        self.assertEqual(converted_id_value, 1234)
        self.assertEqual(converted_year_value, 3000)
        # A non compulsory value when invalid is replaced by an empty string
        self.assertEqual(converted_semimajor_value, EqEntryReader.EMPTY_STRING)

    def test_check_positive_value(self):
        compulsory_field_name = 'depth'
        non_compulsory_field_name = 'sigmaMs'
        compulsory_field_value = -5
        non_compulsory_field_value = -2
        eq_entry = {compulsory_field_name: compulsory_field_value,
                non_compulsory_field_name: non_compulsory_field_value}
        self.eq_reader.check_positive_value(
        non_compulsory_field_name, eq_entry)

        self.assertFalse(self.eq_reader.check_positive_value(
            compulsory_field_name, eq_entry))
        self.assertEqual(EqEntryReader.EMPTY_STRING, eq_entry['sigmaMs'])

    def test_check_year(self):
        field_name = 'year'
        invalid_year = 22015
        eq_entry = {field_name: invalid_year}

        self.assertFalse(self.eq_reader.check_year(field_name, eq_entry))

    def test_check_month(self):
        field_name = 'month'
        invalid_month = 0
        eq_entry = {field_name: invalid_month}

        self.assertFalse(self.eq_reader.check_month(field_name, eq_entry))

    def test_check_day(self):
        field_name = 'day'
        invalid_february_day = 30
        eq_entry = {'month': 2, 'day': invalid_february_day}

        self.assertFalse(self.eq_reader.check_day(field_name, eq_entry))

    def test_check_hour(self):
        field_name = 'hour'
        invalid_hour = 24
        eq_entry = {field_name: invalid_hour}

        self.assertFalse(self.eq_reader.check_hour(field_name, eq_entry))

    def test_check_minute(self):
        field_name = 'minute'
        invalid_minute = 60
        eq_entry = {field_name: invalid_minute}

        self.assertFalse(self.eq_reader.check_minute(field_name, eq_entry))

    def test_check_second(self):
        field_name = 'second'
        invalid_second = -4
        eq_entry = {field_name: invalid_second}
        self.eq_reader.check_second(field_name, eq_entry)

        self.assertEqual(eq_entry['second'], EqEntryReader.EMPTY_STRING)

    def test_check_longitude(self):
        field_name = 'longitude'
        invalid_longitude = -181
        eq_entry = {field_name: invalid_longitude}

        self.assertFalse(self.eq_reader.check_longitude(field_name, eq_entry))

    def test_check_latitude(self):
        field_name = 'latitude'
        invalid_latitude = 91
        eq_entry = {field_name: invalid_latitude}

        self.assertFalse(self.eq_reader.check_latitude(field_name, eq_entry))

    def test_check_epicentre_error_location(self):
        field_name = 'ErrorStrike'
        eq_entry = {field_name: 45, 'SemiMajor90': 5,
            'SemiMinor90': EqEntryReader.EMPTY_STRING}
        self.eq_reader.check_epicentre_error_location(
            field_name, eq_entry)

        self.assertEqual(eq_entry['SemiMajor90'],
            EqEntryReader.EMPTY_STRING)
        self.assertEqual(eq_entry['SemiMinor90'],
            EqEntryReader.EMPTY_STRING)
        self.assertEqual(eq_entry['ErrorStrike'],
            EqEntryReader.EMPTY_STRING)

        eq_entry['SemiMajor90'] = 5
        eq_entry['SemiMinor90'] = 4
        self.eq_reader.check_epicentre_error_location(
            field_name, eq_entry)

        self.assertEqual(eq_entry['SemiMajor90'],
            EqEntryReader.EMPTY_STRING)
        self.assertEqual(eq_entry['SemiMinor90'],
            EqEntryReader.EMPTY_STRING)
        self.assertEqual(eq_entry['ErrorStrike'],
            EqEntryReader.EMPTY_STRING)

    def test_check_sigma_mw(self):
        field_name = 'sigmaMw'
        eq_entry = {field_name: EqEntryReader.EMPTY_STRING}
        self.eq_reader.check_sigma_mw(field_name, eq_entry)

        self.assertEqual(0.0, eq_entry[field_name])

        eq_entry = {field_name: - 0.34}
        self.eq_reader.check_sigma_mw(field_name, eq_entry)

        self.assertEqual(0.0, eq_entry[field_name])


class EqEntryWriterTestCase(unittest.TestCase):

    def setUp(self):
        self.pprocessing_result_filename = get_data_path('out.csv', DATA_DIR)

        self.first_data_row = {'eventID': 1, 'Agency': 'AAA', 'month': 1,
                                'depthError': 0.5, 'second': 13.0,
                                'SemiMajor90': 2.43, 'year': 2000,
                                'ErrorStrike': 298.0, 'timeError': 0.02,
                                'sigmamb': '', 'latitude': 44.368,
                                'sigmaMw': 0.355, 'sigmaMs': '',
                                'Mw': 1.71, 'Ms': '',
                                'Identifier': 20000102034913, 'day': 2,
                                'minute': 49, 'hour': 3,
                                'mb': '', 'SemiMinor90': 1.01,
                                'longitude': 7.282, 'depth': 9.3,
                                'ML': 1.7, 'sigmaML': 0.1}

        self.second_data_row = {'eventID': 2, 'Agency': 'AAA', 'month': 1,
                                'depthError': 0.5, 'second': 57.0,
                                'SemiMajor90': 0.77, 'year': 2000,
                                'ErrorStrike': 315.0, 'timeError': 0.1,
                                'sigmamb': 0.1, 'latitude': 44.318,
                                'sigmaMw': 0.199, 'sigmaMs': '',
                                'Mw': 3.89, 'Ms': '',
                                'Identifier': 20000105132157, 'day': 5,
                                'minute': 21, 'hour': 13,
                                'mb': 3.8, 'SemiMinor90': 0.25,
                                'longitude': 11.988, 'depth': 7.9,
                                'ML': '', 'sigmaML': ''}

        self.writer = EqEntryWriter(self.pprocessing_result_filename)

        self.expected_csv = get_data_path('expected_entries.csv', DATA_DIR)

    def test_an_incorrect_csv_dirname_raise_exception(self):
        self.assertRaises(IOError, EqEntryWriter, 'invalid/dir/name')

    def test_write_csv_file(self):
        rows = [self.first_data_row, self.second_data_row]
        self.writer.write_rows(rows)

        self.assertTrue(filecmp.cmp(self.expected_csv,
            self.pprocessing_result_filename))
