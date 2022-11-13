from swagger_server.models.compute_delivery_information_response import ComputeDeliveryInformationResponse  # noqa: E501
from swagger_server.models.error_response import ErrorResponse  # noqa: E501

from swagger_server.models.area_finder import *

import cv2 as cv
import numpy as np
import urllib.request
import urllib.error


def get_image(hostname="", id="1"):
    if hostname == "":

        return cv.imread("./models/test_image.jpg")

        # url_ = "https://i.postimg.cc/vbFq0CLq/new-objective-4.jpg?dl=1"
        # url_response = urllib.request.urlopen(url_)
        # img_ = cv.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)
        # return img_

    else:
        url_ = "http://" + hostname + "/capture"
        url_response = urllib.request.urlopen(url_)
        return cv.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)


def a_i_service_compute_delivery_information_get(storage_id, camera_hostname, camera_id):  # noqa: E501
    """request AI to compute area of given four woodpieces image

    Request AI to compute # noqa: E501

    :param storage_id: ID of the position of the workpiece to compute delivery information.
    :type storage_id: str
    :param camera_hostname: host name of the camera to get the image from.
    :type camera_hostname: str
    :param camera_id: ID of the required camara on the addressed host. Ignored, as long as only one camera is there.
    :type camera_id: str

    :rtype: ComputeDeliveryInformationResponse
    """
    try:
        assert 0 < np.abs(int(storage_id)) < 5, (405, "Invalid Storage ID")
        img = get_image(camera_hostname, camera_id)
        if int(storage_id) < 0:  # for testing only
            storage_id = str(-1 * int(storage_id))
            img_response = find_area(storage_id, img, return_image=True)
            _, img_encoded = cv.imencode('.jpg', img_response)
            import io
            io_buf = io.BytesIO(img_encoded)
            from flask import send_file
            return send_file(
                io_buf,
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename='0.jpg')
        else:
            x_mm, y_mm, radius_mm, confidence = find_area(storage_id, img)

            response = {
                "xcoordinate": x_mm,
                "ycoordinate": y_mm,
                "radius-mm": radius_mm,
                "confidence": confidence
            }
            return ComputeDeliveryInformationResponse.from_dict(dikt=response)
    except AssertionError as e:
        response = {
            'error_code': e.args[0][0],
            'error_description': e.args[0][1]
        }
        # return ErrorResponse.from_dict(dikt=response)
        return ErrorResponse(error_code=response['error_code'], error_description=response['error_description'])
    except ValueError:
        response = {
            'error_code': 400,
            'error_description': 'Bad Storage ID'
        }
        return ErrorResponse(error_code=response['error_code'], error_description=response['error_description'])
    except urllib.error.URLError as e:
        response = {
            'error_code': 400,
            'error_description': 'Bad Camera Hostname'
        }

        # return ErrorResponse.from_dict(dikt=response)
        return ErrorResponse(error_code=response['error_code'], error_description=response['error_description'])

