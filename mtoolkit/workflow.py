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
to process a series of jobs in a sequential
order. The order is determined by the queue of jobs.
"""

import abc

import yaml

from mtoolkit.jobs import (gardner_knopoff, afteran,
                            stepp, recurrence,
                            read_eq_catalog, read_source_model,
                            create_default_source_model,
                            create_catalog_matrix,
                            create_default_values,
                            create_selected_eq_vector,
                            store_preprocessed_catalog,
                            store_completeness_table,
                            retrieve_completeness_table,
                            maximum_magnitude)

from mtoolkit.scientific.completeness import (stepp_analysis,
                                                selected_eq_flag_vector)

from mtoolkit.scientific.declustering import (gardner_knopoff_decluster,
                                                afteran_decluster)

from mtoolkit.scientific.recurrence import recurrence_analysis

from mtoolkit.scientific.maximum_magnitude import maximum_magnitude_analysis


class PipeLine(object):
    """
    PipeLine allows to create a queue of
    jobs and execute them in order.
    """

    def __init__(self, jobs_list=None):
        """
        Initialize a PipeLine object having
        attributes: name and jobs, a list
        of callable objects.
        """

        self.jobs = []
        if jobs_list != None:
            self.jobs = jobs_list

    def __eq__(self, other):
        """Equal operator for pipeline"""

        return self.jobs == other.jobs

    def __ne__(self, other):
        """Not equal operator for pipeline"""

        return not self.__eq__(other)

    def add_job(self, a_job):
        """Append a new job the to queue"""

        self.jobs.append(a_job)

    def run(self, context):
        """
        Run all the jobs in queue,
        where each job take input data
        and write the results
        of calculation in context.
        If logging is triggered by cmdline
        each job is decorated by adding
        logging statements.
        """

        for job in self.jobs:
            job(context)


class PipeLineBuilder(object):
    """
    PipeLineBuilder allows to build a PipeLine
    by assembling all the required jobs
    specified in the config file.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.map_job_callable = {'GardnerKnopoff': gardner_knopoff,
                                 'Afteran': afteran,
                                 'Stepp': stepp,
                                 'Recurrence': recurrence,
                                 'Create_eq_vector':
                                   create_selected_eq_vector,
                                 'Store_eq_catalog':
                                   store_preprocessed_catalog,
                                 'Store_completeness_table':
                                   store_completeness_table,
                                 'Retrieve_completeness_table':
                                   retrieve_completeness_table,
                                 'MaximumMagnitude':
                                   maximum_magnitude}

    @abc.abstractmethod
    def build(self, config):
        """
        Build method creates the pipeline by
        assembling all the steps required.
        The steps described in the config
        could be preprocessing or processing
        steps
        """

    def append_jobs(self, pipeline, jobs):
        """
        Add jobs to the pipeline by looking
        at the list of jobs defined in the
        config
        """

        for job in jobs:
            if job in self.map_job_callable:
                pipeline.add_job(self.map_job_callable[job])
            else:
                raise RuntimeError('Invalid job: %s' % job)

        return pipeline


class PreprocessingBuilder(PipeLineBuilder):
    """
    PreprocessingBuilder allows to build a
    preprocessing pipeLine
    """

    PREPROCESSING_JOBS_KEY = 'preprocessing_jobs'
    PPROCESSING_RESULT_KEY = 'pprocessing_result_file'

    def build(self, config):

        # Checks if source model is defined
        if config['source_model_file']:
            source_model_creation = read_source_model
        else:
            source_model_creation = create_default_source_model

        # Add compulsory jobs to the pipeline'])
        pipeline = PipeLine([read_eq_catalog, source_model_creation,
                    create_catalog_matrix, create_default_values])

        # Add preprocessing jobs
        if config[PreprocessingBuilder.PREPROCESSING_JOBS_KEY]:
            self.append_jobs(pipeline,
                    config[PreprocessingBuilder.PREPROCESSING_JOBS_KEY])

            # Add store eq catalog jobs if result file is defined
            if config[PreprocessingBuilder.PPROCESSING_RESULT_KEY]:
                pipeline.add_job(self.map_job_callable['Create_eq_vector'])
                pipeline.add_job(self.map_job_callable['Store_eq_catalog'])
                pipeline.add_job(
                    self.map_job_callable['Store_completeness_table'])
        else:
            if config['completeness_table_file']:
                pipeline.add_job(
                    self.map_job_callable['Retrieve_completeness_table'])

        return pipeline


class ProcessingBuilder(PipeLineBuilder):
    """
    ProcessingBuilder allows to build a
    processing pipeLine
    """

    PROCESSING_JOBS_CONFIG_KEY = 'processing_jobs'

    def build(self, config):
        pipeline = PipeLine()

        if config[ProcessingBuilder.PROCESSING_JOBS_CONFIG_KEY]:
            self.append_jobs(pipeline,
                    config[ProcessingBuilder.PROCESSING_JOBS_CONFIG_KEY])

        return pipeline


class Context(object):
    """
    Context allows to read the config file
    and store preprocessing/processing steps
    intermediate results.
    """

    def __init__(self, config_filename=None):
        self.config = dict()
        self.map_sc = {'gardner_knopoff': gardner_knopoff_decluster,
                        'afteran': afteran_decluster,
                        'stepp': stepp_analysis,
                        'recurrence': recurrence_analysis,
                        'select_eq_vector': selected_eq_flag_vector,
                        'maximum_magnitude': maximum_magnitude_analysis}

        if config_filename:
            config_file = open(config_filename, 'r')
            self.config = yaml.load(config_file)

        self.eq_catalog = None
        self.sm_definitions = None
        self.catalog_matrix = None
        self.working_catalog = None
        self.completeness_table = None


class Workflow(object):
    """
    Workflow is the object responsible
    for dealing with preprocessing and
    processing pipelines
    """

    def __init__(self, preprocessing_pipeline, processing_pipeline):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.processing_pipeline = processing_pipeline

    def start(self, context, catalog_filter):
        """
        Execute the main workflow
        """
        self.preprocessing_pipeline.run(context)
        if context.config['apply_processing_jobs']:
            for sm, filtered_eq in catalog_filter.filter_eqs(
                    context.sm_definitions, context.working_catalog):

                context.cur_sm = sm
                context.current_filtered_eq = filtered_eq
                self.processing_pipeline.run(context)
