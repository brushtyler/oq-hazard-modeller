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
import os
from lxml import etree

from nrml.nrml_xml import get_data_path, DATA_DIR, SCHEMA_DIR

from nrml.reader import NRMLReader

from nrml.writer import AreaSourceWriter

from mtoolkit.source_model import (AreaSource, POINT, AREA_BOUNDARY,
                                    TRUNCATED_GUTEN_RICHTER)

from mtoolkit.source_model import (MAGNITUDE, RUPTURE_RATE_MODEL,
                                    RUPTURE_DEPTH_DISTRIB)

AREA_SOURCE = get_data_path('area_source_model.xml', DATA_DIR)
AREA_SOURCES = get_data_path('area_sources.xml', DATA_DIR)
INCORRECT_NRML = get_data_path('incorrect_area_source_model.xml', DATA_DIR)
SCHEMA = get_data_path('nrml.xsd', SCHEMA_DIR)

OUTPUT_NRML = os.path.join(
    get_data_path('', DATA_DIR), 'serialized_models.xml')


def create_area_source():

    asource = AreaSource()
    asource.nrml_id = "n1"
    asource.source_model_id = "sm1"
    asource.area_source_id = "src03"
    asource.name = "Quito"
    asource.tectonic_region = "Active Shallow Crust"

    area_boundary = [-122.5, 37.5, -121.5,
                        37.5, -121.5, 38.5,
                        -122.5, 38.5]
    asource.area_boundary = AREA_BOUNDARY("urn:ogc:def:crs:EPSG::4326",
            [POINT(area_boundary[i],
                area_boundary[i + 1])
                for i in xrange(0, len(area_boundary), 2)])

    truncated_gutenberg_richter = TRUNCATED_GUTEN_RICHTER(
        5.0, 0.8, 5.0, 7.0, "ML")

    strike = 0.0
    dip = 90.0
    rake = 0.0

    asource.rupture_rate_model = RUPTURE_RATE_MODEL(
        truncated_gutenberg_richter, strike, dip, rake)

    magnitude = MAGNITUDE("ML", [6.0, 6.5, 7.0])
    depth = [5000.0, 3000.0, 0.0]

    asource.rupture_depth_dist = RUPTURE_DEPTH_DISTRIB(magnitude, depth)

    asource.hypocentral_depth = 5000.0

    return asource


class NRMLReaderTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_read_area_source_model(self):
        area_source = create_area_source()
        as_reader = NRMLReader(AREA_SOURCE, SCHEMA)
        area_source_gen = as_reader.read()
        read_first_area_source = area_source_gen.next()

        self.assertEqual(area_source, read_first_area_source)

    def test_generated_area_sources(self):
        num_expected_area_sources = 2
        as_reader = NRMLReader(AREA_SOURCES, SCHEMA)
        for num_area_sources, _ in enumerate(as_reader.read(), start=1):
            pass
        self.assertEqual(num_expected_area_sources, num_area_sources)


class AreaSourceWriterTestCase(unittest.TestCase):

    def setUp(self):
        self.as_writer = AreaSourceWriter(OUTPUT_NRML)
        self.area_source = create_area_source()

    def test_serialize_area_source_definition(self):
        self.as_writer.serialize([self.area_source])

        xml_tree_written_file = etree.parse(open(OUTPUT_NRML))
        xml_tree_expected_file = etree.parse(open(AREA_SOURCE))

        self.assertEqual(etree.tostring(
                            xml_tree_written_file, pretty_print=True),
                etree.tostring(xml_tree_expected_file, pretty_print=True))

    def test_writer_creates_valid_nrml(self):
        self.as_writer.serialize([self.area_source])

        xml_doc = etree.parse(OUTPUT_NRML)
        xml_schema = etree.XMLSchema(etree.parse(SCHEMA))

        self.assertTrue(xml_schema.validate(xml_doc))
