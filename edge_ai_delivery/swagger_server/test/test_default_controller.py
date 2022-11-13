# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.compute_delivery_information_response import ComputeDeliveryInformationResponse  # noqa: E501
from swagger_server.models.error_response import ErrorResponse  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_a_i_service_compute_delivery_information_get(self):
        """Test case for a_i_service_compute_delivery_information_get

        request AI to compute area of given four woodpieces image
        """
        query_string = [('storage_id', 'storage_id_example'),
                        ('camera_hostname', 'camera_hostname_example'),
                        ('camera_id', 'camera_id_example')]
        response = self.client.open(
            '/AI_Service/compute_deliveryInformation',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
