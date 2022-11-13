# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class ComputeDeliveryInformationResponseInner(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, xcoordinate: float=None, ycoordinate: float=None, radius_mm: float=None, confidence: float=None):  # noqa: E501
        """ComputeDeliveryInformationResponseInner - a model defined in Swagger

        :param xcoordinate: The xcoordinate of this ComputeDeliveryInformationResponseInner.  # noqa: E501
        :type xcoordinate: float
        :param ycoordinate: The ycoordinate of this ComputeDeliveryInformationResponseInner.  # noqa: E501
        :type ycoordinate: float
        :param radius_mm: The radius_mm of this ComputeDeliveryInformationResponseInner.  # noqa: E501
        :type radius_mm: float
        :param confidence: The confidence of this ComputeDeliveryInformationResponseInner.  # noqa: E501
        :type confidence: float
        """
        self.swagger_types = {
            'xcoordinate': float,
            'ycoordinate': float,
            'radius_mm': float,
            'confidence': float
        }

        self.attribute_map = {
            'xcoordinate': 'xcoordinate',
            'ycoordinate': 'ycoordinate',
            'radius_mm': 'radius-mm',
            'confidence': 'confidence'
        }
        self._xcoordinate = xcoordinate
        self._ycoordinate = ycoordinate
        self._radius_mm = radius_mm
        self._confidence = confidence

    @classmethod
    def from_dict(cls, dikt) -> 'ComputeDeliveryInformationResponseInner':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The ComputeDeliveryInformationResponse_inner of this ComputeDeliveryInformationResponseInner.  # noqa: E501
        :rtype: ComputeDeliveryInformationResponseInner
        """
        return util.deserialize_model(dikt, cls)

    @property
    def xcoordinate(self) -> float:
        """Gets the xcoordinate of this ComputeDeliveryInformationResponseInner.

        measured from the upper left marker (the most left for all storage places)  # noqa: E501

        :return: The xcoordinate of this ComputeDeliveryInformationResponseInner.
        :rtype: float
        """
        return self._xcoordinate

    @xcoordinate.setter
    def xcoordinate(self, xcoordinate: float):
        """Sets the xcoordinate of this ComputeDeliveryInformationResponseInner.

        measured from the upper left marker (the most left for all storage places)  # noqa: E501

        :param xcoordinate: The xcoordinate of this ComputeDeliveryInformationResponseInner.
        :type xcoordinate: float
        """

        self._xcoordinate = xcoordinate

    @property
    def ycoordinate(self) -> float:
        """Gets the ycoordinate of this ComputeDeliveryInformationResponseInner.

        measured from the upper left marker  # noqa: E501

        :return: The ycoordinate of this ComputeDeliveryInformationResponseInner.
        :rtype: float
        """
        return self._ycoordinate

    @ycoordinate.setter
    def ycoordinate(self, ycoordinate: float):
        """Sets the ycoordinate of this ComputeDeliveryInformationResponseInner.

        measured from the upper left marker  # noqa: E501

        :param ycoordinate: The ycoordinate of this ComputeDeliveryInformationResponseInner.
        :type ycoordinate: float
        """

        self._ycoordinate = ycoordinate

    @property
    def radius_mm(self) -> float:
        """Gets the radius_mm of this ComputeDeliveryInformationResponseInner.


        :return: The radius_mm of this ComputeDeliveryInformationResponseInner.
        :rtype: float
        """
        return self._radius_mm

    @radius_mm.setter
    def radius_mm(self, radius_mm: float):
        """Sets the radius_mm of this ComputeDeliveryInformationResponseInner.


        :param radius_mm: The radius_mm of this ComputeDeliveryInformationResponseInner.
        :type radius_mm: float
        """

        self._radius_mm = radius_mm

    @property
    def confidence(self) -> float:
        """Gets the confidence of this ComputeDeliveryInformationResponseInner.


        :return: The confidence of this ComputeDeliveryInformationResponseInner.
        :rtype: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, confidence: float):
        """Sets the confidence of this ComputeDeliveryInformationResponseInner.


        :param confidence: The confidence of this ComputeDeliveryInformationResponseInner.
        :type confidence: float
        """

        self._confidence = confidence
