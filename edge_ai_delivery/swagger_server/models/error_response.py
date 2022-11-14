# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class ErrorResponse(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, error_code: int=None, error_description: str=None):  # noqa: E501
        """ErrorResponse - a model defined in Swagger

        :param error_code: The error_code of this ErrorResponse.  # noqa: E501
        :type error_code: int
        :param error_description: The error_description of this ErrorResponse.  # noqa: E501
        :type error_description: str
        """
        self.swagger_types = {
            'error_code': int,
            'error_description': str
        }

        self.attribute_map = {
            'error_code': 'error-code',
            'error_description': 'error-description'
        }
        self._error_code = error_code
        self._error_description = error_description

    @classmethod
    def from_dict(cls, dikt) -> 'ErrorResponse':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The ErrorResponse of this ErrorResponse.  # noqa: E501
        :rtype: ErrorResponse
        """
        return util.deserialize_model(dikt, cls)

    @property
    def error_code(self) -> int:
        """Gets the error_code of this ErrorResponse.


        :return: The error_code of this ErrorResponse.
        :rtype: int
        """
        return self._error_code

    @error_code.setter
    def error_code(self, error_code: int):
        """Sets the error_code of this ErrorResponse.


        :param error_code: The error_code of this ErrorResponse.
        :type error_code: int
        """

        self._error_code = error_code

    @property
    def error_description(self) -> str:
        """Gets the error_description of this ErrorResponse.


        :return: The error_description of this ErrorResponse.
        :rtype: str
        """
        return self._error_description

    @error_description.setter
    def error_description(self, error_description: str):
        """Sets the error_description of this ErrorResponse.


        :param error_description: The error_description of this ErrorResponse.
        :type error_description: str
        """

        self._error_description = error_description