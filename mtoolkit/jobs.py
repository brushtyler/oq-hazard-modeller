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
The purpose of this module is to provide functions which tackle specific job,
some of them wrap scientific functions defined in the scientific module.
"""

import logging
import numpy as np

from mtoolkit.eqcatalog import EqEntryReader, EqEntryWriter
from nrml.reader import NRMLReader
from nrml.nrml_xml import get_data_path, SCHEMA_DIR
from mtoolkit.source_model import default_area_source


NRML_SCHEMA_PATH = get_data_path('nrml.xsd', SCHEMA_DIR)
CATALOG_COMPLETENESS_MATRIX_YEAR_INDEX = 0
CATALOG_MATRIX_MW_INDEX = 5
CATALOG_MATRIX_FIXED_COLOUMNS = ['year', 'month', 'day',
                                'longitude', 'latitude', 'Mw', 'sigmaMw']
COMPLETENESS_TABLE_MW_INDEX = 1
SIGMA_MW_INDEX = 6

LOGGER = logging.getLogger('mt_logger')


def logged_job(job):
    """
    Decorate a job by adding logging
    statements before and after the execution
    of the job.
    """

    def wrapper(context):
        """Wraps a job, adding logging statements"""
        LOGGER.info(''.center(80, '-'))
        LOGGER.info(" %22s " % job.__name__.upper())
        job(context)

    return wrapper


@logged_job
def read_eq_catalog(context):
    """
    Create eq entries by reading an e] catalog.
    :param context: shared datastore across different jobs
        in a pipeline
    """

    with open(context.config['eq_catalog_file']) as eq_catalog:
        reader = EqEntryReader(eq_catalog)
        context.eq_catalog = reader.read_eq_catalog()

    LOGGER.debug("* Eq catalog length: %s" % len(context.eq_catalog))


@logged_job
def read_source_model(context):
    """
    Create source model definitions by reading a source model.
    :param context: shared datastore across different jobs
        in a pipeline
    """

    sm_definitions = []

    reader = NRMLReader(context.config['source_model_file'], NRML_SCHEMA_PATH)
    for sm in reader.read():
        sm_definitions.append(sm)

        context.sm_definitions = sm_definitions

    LOGGER.debug("* Eq number source models: %s" % len(context.sm_definitions))


@logged_job
def create_default_source_model(context):
    """
    Create a default source model object
    :param context: shared datastore across different jobs
        in a pipeline
    """

    context.sm_definitions = [default_area_source()]

    LOGGER.debug("* Eq number source models: %s" % len(context.sm_definitions))


@logged_job
def create_catalog_matrix(context):
    """
    Create a numpy matrix according to fixed attributes.
    :param context: shared datastore across different jobs
        in a pipeline
    """

    matrix = []
    for eq_entry in context.eq_catalog:
        matrix.append([eq_entry[coloumn] for coloumn in
                        CATALOG_MATRIX_FIXED_COLOUMNS])

    context.catalog_matrix = np.array(matrix)
    context.working_catalog = np.array(matrix)


@logged_job
def create_default_values(context):
    """
    Create default values for context attributes to be
    used in different kinds of workflows.
    :param context: shared datastore across different jobs
        in a pipeline
    """

    context.flag_vector = np.zeros(len(context.working_catalog))
    min_year = context.working_catalog[:,
        CATALOG_COMPLETENESS_MATRIX_YEAR_INDEX].min()
    min_magnitude = context.working_catalog[:, CATALOG_MATRIX_MW_INDEX].min()
    context.completeness_table = np.array([[min_year, min_magnitude]])


@logged_job
def gardner_knopoff(context):
    """
    Apply gardner_knopoff declustering algorithm to the eq catalog.
    :param context: shared datastore across different jobs
        in a pipeline
    """

    vcl, vmain_shock, flag_vector = context.map_sc['gardner_knopoff'](
            context.working_catalog,
            context.config['GardnerKnopoff']['time_dist_windows'],
            context.config['GardnerKnopoff']['foreshock_time_window'])

    context.vcl = vcl
    context.working_catalog = vmain_shock
    context.flag_vector = flag_vector

    LOGGER.debug(
        "* Number of events after declustering: %s" % len(vmain_shock))

    LOGGER.debug(
        "* Number of events removed during declustering: %s" %
        (np.sum(flag_vector != 0)))

    LOGGER.debug(
        "* Number of clusters identified: %s" %
        (np.size(np.unique(vcl), 0) - 1))


@logged_job
def afteran(context):
    """
    Apply afteran declustering algorithm to the eq catalog.
    :param context: shared datastore across different jobs
        in a pipeline
    """

    vcl, vmain_shock, flag_vector = context.map_sc['afteran'](
            context.catalog_matrix,
            context.config['Afteran']['time_dist_windows'],
            context.config['Afteran']['time_window'])

    context.vcl = vcl
    context.working_catalog = vmain_shock
    context.flag_vector = flag_vector

    LOGGER.debug(
        "* Number of events after declustering: %s" % len(vmain_shock))

    LOGGER.debug(
        "* Number of events removed during declustering: %s" %
        (np.sum(flag_vector != 0)))

    LOGGER.debug(
        "* Number of clusters identified: %s" %
        (np.size(np.unique(vcl), 0) - 1))


@logged_job
def stepp(context):
    """
    Apply step algorithm to the catalog matrix
    :param context: shared datastore across different jobs
        in a pipeline
    """

    context.completeness_table = context.map_sc['stepp'](
        context.working_catalog[:, CATALOG_COMPLETENESS_MATRIX_YEAR_INDEX],
        context.working_catalog[:, CATALOG_MATRIX_MW_INDEX],
        context.config['Stepp']['magnitude_windows'],
        context.config['Stepp']['time_window'],
        context.config['Stepp']['sensitivity'],
        context.config['Stepp']['increment_lock'])

    LOGGER.debug(
        "* Number of events into completeness algorithm: %s"
            % len(context.working_catalog))

    LOGGER.debug(
        "* Completeness table: ")

    LOGGER.debug(context.completeness_table)


@logged_job
def create_selected_eq_vector(context):
    """
    Apply selected_eq_flag_vector algorithm to
    the catalog matrix and completeness table
    :param context: shared datastore across different jobs
        in a pipeline
    """

    context.selected_eq_vector = context.map_sc['select_eq_vector'](
        context.catalog_matrix[:, CATALOG_COMPLETENESS_MATRIX_YEAR_INDEX],
        context.catalog_matrix[:, CATALOG_MATRIX_MW_INDEX],
        context.completeness_table[:, CATALOG_COMPLETENESS_MATRIX_YEAR_INDEX],
        context.completeness_table[:, COMPLETENESS_TABLE_MW_INDEX],
        context.flag_vector)


@logged_job
def store_preprocessed_catalog(context):
    """
    Write in a csv file the earthquake
    catalog after preprocessing jobs (i.e.
    gardner_knopoff, stepp)
    :param context: shared datastore across different jobs
        in a pipeline
    """

    writer = EqEntryWriter(
        context.config['pprocessing_result_file'])
    indexes_entries_to_store = np.where(context.selected_eq_vector == 0)[0]
    number_written_eq = len(indexes_entries_to_store)

    entries = []
    for index in indexes_entries_to_store:
        entries.append(context.eq_catalog[index])

    writer.write_rows(entries)

    LOGGER.debug("* Stored Eq entries: %d" % number_written_eq)

    LOGGER.debug("* Number of events removed after preprocessing jobs: %d" %
        (len(context.catalog_matrix) - number_written_eq))


@logged_job
def store_completeness_table(context):
    """
    Store in a csv file the completeness
    table after preprocessing jobs (i.e.
    gardner_knopoff, stepp)
    :param context: shared datastore across different jobs
        in a pipeline
    """

    np.savetxt(context.config['completeness_table_file'],
        context.completeness_table, fmt='%10.5f', delimiter=',')

    LOGGER.debug("* Completeness Table stored")


@logged_job
def retrieve_completeness_table(context):
    """
    Retrieve from a csv file the completeness
    table
    :param context: shared datastore across different jobs
        in a pipeline
    """

    context.completeness_table = np.genfromtxt(
        context.config['completeness_table_file'], delimiter=',').reshape(
            (-1, 2))

    LOGGER.debug("* Completeness Table retrieved")


@logged_job
def recurrence(context):
    """
    Apply recurrence algorithm to the filtered catalog
    matrix and completeness table
    :param context: shared datastore across different jobs
        in a pipeline
    """

    bval, sigb, a_m, siga_m = context.map_sc['recurrence'](
            context.current_filtered_eq[:,
                CATALOG_COMPLETENESS_MATRIX_YEAR_INDEX],
            context.current_filtered_eq[:, CATALOG_MATRIX_MW_INDEX],
            context.completeness_table,
            context.config['Recurrence']['magnitude_window'],
            context.config['Recurrence']['recurrence_algorithm'],
            context.config['Recurrence']['reference_magnitude'],
            context.config['Recurrence']['time_window'])

    t = context.cur_sm.rupture_rate_model.truncated_gutenberg_richter._replace(
        a_value=a_m,
        b_value=bval,
        min_magnitude=context.config['Recurrence']['reference_magnitude'])

    context.cur_sm.rupture_rate_model = \
        context.cur_sm.rupture_rate_model._replace(
                                        truncated_gutenberg_richter=t)

    context.cur_sm.recurrence_sigb = sigb
    context.cur_sm.recurrence_siga_m = siga_m

    LOGGER.debug("Bvalue: %3.3f, Sigma_b: %3.3f, Avalue: %3.3f, Sigma_a: %3.3f"
        % (bval, sigb, a_m, siga_m))


@logged_job
def maximum_magnitude(context):
    """
    Apply maximum magnitude algorithm to the filtered catalog
    matrix, completeness table, bvalue and bvalue uncertainty,
    this job should depends on value computed by recurrence.
    :param context: shared datastore across different jobs
        in a pipeline
    """
    max_mag, max_mag_sigma = context.map_sc['maximum_magnitude'](
        context.current_filtered_eq[:,
            CATALOG_COMPLETENESS_MATRIX_YEAR_INDEX],
        context.current_filtered_eq[:, CATALOG_MATRIX_MW_INDEX],
        context.current_filtered_eq[:, SIGMA_MW_INDEX],
        context.config['MaximumMagnitude']['maxim_mag_algorithm'],
        context.config['MaximumMagnitude']['iteration_tolerance'],
        context.config['MaximumMagnitude']['maximum_iterations'],
        context.config['MaximumMagnitude']['neq'],
        context.config['MaximumMagnitude']['number_samples'],
        context.config['MaximumMagnitude']['number_bootstraps'])

    t = context.cur_sm.rupture_rate_model.truncated_gutenberg_richter._replace(
        max_magnitude=max_mag)

    context.cur_sm.rupture_rate_model = \
        context.cur_sm.rupture_rate_model._replace(
                                        truncated_gutenberg_richter=t)

    context.cur_sm.max_mag_sigma = max_mag_sigma

    LOGGER.debug("Max magnitude: %3.3f, Sigma: %3.3f"
        % (max_mag, max_mag_sigma))
