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


import unittest
import numpy as np

from mtoolkit.workflow import Context, PipeLineBuilder, Workflow
from mtoolkit.jobs import (read_eq_catalog, read_source_model,
                           create_catalog_matrix, gardner_knopoff,
                           recurrence, create_default_values)
from mtoolkit.catalog_filter import SourceModelCatalogFilter
from mtoolkit.utils import get_data_path, DATA_DIR

DECIMAL_PLACES = 5
RUPTURE_KEY = 'rupture_rate_model'


def create_workflow(config):
    builder = PipeLineBuilder()
    preprocessing_pipeline = builder.build(config,
        PipeLineBuilder.PREPROCESSING_JOBS_CONFIG_KEY,
         [read_eq_catalog, read_source_model,
            create_catalog_matrix, create_default_values])

    processing_pipeline = builder.build(config,
        PipeLineBuilder.PROCESSING_JOBS_CONFIG_KEY)

    return Workflow(preprocessing_pipeline, processing_pipeline)


def create_context(filename=None):
    return Context(get_data_path(filename, DATA_DIR))


def run(workflow, context):
    return workflow.start(context, SourceModelCatalogFilter())


class JobsTestCase(unittest.TestCase):

    def setUp(self):
        self.context = create_context('config_processing.yml')

        self.eq_catalog_filename = get_data_path(
            'ISC_small_data.csv', DATA_DIR)
        self.smodel_filename = get_data_path(
            'area_source_model.xml', DATA_DIR)

    def test_read_eq_catalog(self):
        self.context.config['eq_catalog_file'] = self.eq_catalog_filename
        expected_first_eq_entry = {'eventID': 1, 'Agency': 'AAA', 'month': 1,
                'depthError': 0.5, 'second': 13.0, 'SemiMajor90': 2.43,
                'year': 2000, 'ErrorStrike': 298.0, 'timeError': 0.02,
                'sigmamb': '', 'latitude': 44.368, 'sigmaMw': 0.355,
                'sigmaMs': '', 'Mw': 1.71, 'Ms': '',
                'Identifier': 20000102034913, 'day': 2, 'minute': 49,
                'hour': 3, 'mb': '', 'SemiMinor90': 1.01, 'longitude': 7.282,
                'depth': 9.3, 'ML': 1.7, 'sigmaML': 0.1}

        read_eq_catalog(self.context)

        self.assertEqual(10, len(self.context.eq_catalog))
        self.assertEqual(expected_first_eq_entry,
                self.context.eq_catalog[0])

    def test_read_smodel(self):
        self.context.config['source_model_file'] = self.smodel_filename
        expected_first_sm_definition = \
            {'id_as': 'src_1',
             'area_boundary':
               [132.93, 42.85, 134.86,
                41.82, 129.73, 38.38,
                128.15, 40.35],
             'rupture_rate_model': [
               {'max_magnitude': 8.0,
                'a_value_cumulative':
                    3.1612436654341836,
                'name': 'truncated_guten_richter',
                'min_magnitude': 5.0,
                'b_value': 0.7318999871612379},
             {'name': 'focal_mechanism',
              'nodal_planes': [
                {'strike': 0.0,
                 'rake': 0.0,
                 'dip': 90.0,
                 'id': 0},
                {'strike': 12.0,
                 'rake': 0.0,
                 'dip': 40.0,
                 'id': 1}],
              'id': 'smi:fm1/0'}],
             'tectonic_region':
                'Active Shallow Crust',
             'id_sm': 'sm1',
             'rupture_depth_distribution': {
                'depth': [15.0],
                'magnitude': [5.0],
                'name': 'rupture_depth_distrib'},
             'hypocentral_depth': 15.0,
             'type': 'area_source',
             'name': 'Source 8.CH.3'}

        read_source_model(self.context)
        self.assertEqual(2, len(self.context.sm_definitions))
        self.assertEqual(expected_first_sm_definition,
                self.context.sm_definitions[0])

    def test_gardner_knopoff(self):
        context = create_context('config_gardner_knopoff.yml')

        workflow = create_workflow(context.config)
        run(workflow, context)

        expected_vmain_shock = np.delete(
            context.catalog_matrix, [4, 10, 19], 0)

        expected_vcl = np.array([0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
            0, 0, 0, 0, 6])

        expected_flag_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1])

        gardner_knopoff(context)

        self.assertTrue(np.array_equal(expected_vmain_shock,
                context.catalog_matrix))
        self.assertTrue(np.array_equal(expected_vcl, context.vcl))
        self.assertTrue(np.array_equal(expected_flag_vector,
                context.flag_vector))

    def test_parameters_gardner_knopoff(self):
        context = create_context('config_gardner_knopoff.yml')

        context.catalog_matrix = []

        def assert_parameters(data, time_dist_windows, foreshock_time_window):
            self.assertEquals("GardnerKnopoff", time_dist_windows)
            self.assertEquals(0.5, foreshock_time_window)
            return [], [], []

        context.map_sc['gardner_knopoff'] = assert_parameters
        gardner_knopoff(context)

    def test_stepp(self):
        context = create_context('config_stepp.yml')

        workflow = create_workflow(context.config)

        run(workflow, context)

        filtered_eq_events = np.array([
                    [1994., 4.0], [1994., 4.1], [1994., 4.2],
                    [1994., 4.3], [1994., 4.4], [1964., 4.5],
                    [1964., 4.6], [1964., 4.7], [1964., 4.8],
                    [1964., 4.9], [1964., 5.0], [1964., 5.1],
                    [1964., 5.2], [1964., 5.3], [1964., 5.4],
                    [1919., 5.5], [1919., 5.6], [1919., 5.7],
                    [1919., 5.8], [1919., 5.9], [1919., 6.0],
                    [1919., 6.1], [1919., 6.2], [1919., 6.3],
                    [1919., 6.4], [1919., 6.5], [1919., 6.6],
                    [1919., 6.7], [1919., 6.8], [1919., 6.9],
                    [1919., 7.0], [1919., 7.1], [1919., 7.2],
                    [1919., 7.3]])

        self.assertTrue(np.allclose(filtered_eq_events,
                context.completeness_table))

    def test_parameters_stepp(self):
        context = create_context('config_stepp.yml')

        workflow = create_workflow(context.config)

        def assert_parameters(year, mw, magnitude_windows, time_window,
                sensitivity, iloc):
            self.assertEqual(time_window, 5)
            self.assertEqual(magnitude_windows, 0.1)
            self.assertEqual(sensitivity, 0.2)
            self.assertTrue(iloc)

        context.map_sc['stepp'] = assert_parameters

        run(workflow, context)

    def test_recurrence_wiechart_algorithm(self):
        context = create_context('config_recurrence_wiechart.yml')
        workflow = create_workflow(context.config)

        run(workflow, context)

        self.assertAlmostEqual(
            context.sm_definitions[0][RUPTURE_KEY][0]['b_value'],
            0.569790,
            DECIMAL_PLACES)

        self.assertAlmostEqual(
            context.sm_definitions[0]['Recurrence_sigb'],
            0.041210,
            DECIMAL_PLACES)

        self.assertAlmostEqual(
            context.sm_definitions[0][RUPTURE_KEY][0]['a_value_cumulative'],
            132.051268,
            DECIMAL_PLACES)

        self.assertAlmostEqual(
            context.sm_definitions[0]['Recurrence_siga_m'],
            7.701386,
            DECIMAL_PLACES)

    def test_recurrence_mle_algorithm(self):
        context = create_context('config_recurrence_mle.yml')

        workflow = create_workflow(context.config)
        run(workflow, context)

        self.assertAlmostEqual(
            context.sm_definitions[0][RUPTURE_KEY][0]['b_value'],
            0.595256,
            DECIMAL_PLACES)

        self.assertAlmostEqual(
            context.sm_definitions[0]['Recurrence_sigb'],
            0.024816,
            DECIMAL_PLACES)

        self.assertAlmostEqual(
            context.sm_definitions[0][RUPTURE_KEY][0]['a_value_cumulative'],
            3.123129,
            DECIMAL_PLACES)

        self.assertAlmostEqual(
            context.sm_definitions[0]['Recurrence_siga_m'],
            0.027298,
            DECIMAL_PLACES)

    def test_parameters_recurrence(self):
        self.context.config['Recurrence']['magnitude_window'] = 0.5
        self.context.config['Recurrence']['recurrence_algorithm'] = 'Wiechart'
        self.context.config['Recurrence']['referece_magnitude'] = 1.1
        self.context.config['Recurrence']['time_window'] = 0.3

        # Fake values for used attributes
        self.context.current_sm = {'rupture_rate_model': [{'max_magnitude': '',
                                    'a_value_cumulative': '',
                                    'name': '',
                                    'min_magnitude': '',
                                    'b_value': ''}]}
        self.context.completeness_table = []
        self.context.current_filtered_eq = np.array([[1, 2, 3, 4, 5, 6]])

        def assert_parameters(year_col, magnitude_col, completeness_table,
            magnitude_window, recurrence_algorithm, reference_magnitude,
            time_window):

            self.assertEqual(magnitude_window, 0.5)
            self.assertEqual(recurrence_algorithm, 'Wiechart')
            self.assertEqual(reference_magnitude, 1.1)
            self.assertEqual(time_window, 0.3)
            return 0.0, 0.0, 0.0, 0.0

        self.context.map_sc['recurrence'] = assert_parameters
        recurrence(self.context)
