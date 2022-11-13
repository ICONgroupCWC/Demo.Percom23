import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt
import math
from swagger_server.models.image_process import test_generator, save_result
from polylabel import polylabel
# from tensorflow import keras

from swagger_server.__init__ import MARKER_MODEL_INIT, SHOW_PLOTS

MARKER_MODEL = MARKER_MODEL_INIT
MARKER_GAP = 280
TABLE_OFFSET = 100


def get_translated_points(points, matrix):
    new_points = []
    for p in points:
        px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / \
             (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
        py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / \
             (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
        new_points.append([int(round(px)), int(round(py))])
    return new_points


def visualize(img, title=None, height=600):
    if SHOW_PLOTS is True:
        width = int(height * img.shape[1] / img.shape[0])
        # print(height, width)
        resized = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
        if title is None:
            title = "Image"
        cv.imshow(title, resized)
        cv.waitKey(0)
        cv.destroyAllWindows()


def pixel_to_mm(length, ratio=1.):
    return length * ratio


def update_markers(points):
    markers = [
        [points[0], points[1], points[8], points[9]],  # corners in order TL, TR, BR, BL
        [points[1], points[2], points[7], points[8]],  # corners in order TL, TR, BR, BL
        [points[2], points[3], points[6], points[7]],  # corners in order TL, TR, BR, BL
        [points[3], points[4], points[5], points[6]]  # corners in order TL, TR, BR, BL
    ]
    return markers


def get_markers(img_input):
    # splitting
    size = 600

    points = [
        [157, 1008], [1055, 1029], [1956, 1044], [2852, 1067], [3736, 1086],
        [3716, 1977], [2834, 1966], [1936, 1950], [1035, 1934], [142, 1907]
    ]
    """
    points = [
        [20, 756], [860, 756], [1700, 756], [2540, 756], [3380, 756],
        [20, 1600], [860, 1600], [1700, 1600], [2540, 1600], [3380, 1600]
    ]   
    """
    img_segments = []
    segment_origin = []
    for idx in range(10):
        img_segments.append(
            img_input[
            max(int(points[idx][1] - size / 2), 0): min(int(points[idx][1] + size / 2), img_input.shape[0]),
            max(int(points[idx][0] - size / 2), 0): min(int(points[idx][0] + size / 2), img_input.shape[1])]
        )
        segment_origin.append(
            [max(int(points[idx][1] - size / 2), 0), max(int(points[idx][0] - size / 2), 0)])
        """
        img_segments.append(
            img_input[points[idx][1]:(points[idx][1] + size), points[idx][0]:(points[idx][0] + size)]
        )
        segment_origin.append([points[idx][1], points[idx][0]])
        """
        """
        filename = "segment_" + str(idx) + ".png"
        matplotlib.image.imsave(filename, img_segments[idx], cmap='gray')
        """
    # visualize
    """
    if GENERATE_IMAGE is True:
        combined_segments = np.vstack((
            np.hstack(tuple(map(tuple, img_segments[:5]))),
            np.hstack(tuple(map(tuple, img_segments[5:])))
        ))
        visualize(combined_segments, title="Original segments")
    """
    # converting images into marker detected images
    """"""
    input_images = test_generator(img_segments)
    output_images = MARKER_MODEL.predict(input_images, len(img_segments), verbose=1)
    images_marker = save_result(output_images, flag_multi_class=False)
    # np.save('marker_images.npy', images_marker)
    """"""
    # extracting location of the centre_ of the markers
    centers_local = []
    centers_global = []
    # images_marker = np.load('marker_images.npy')
    # print(images_marker)
    for indMarkers in range(len(images_marker)):
        segment_input = images_marker[indMarkers]
        segment_input = np.transpose(segment_input)
        segment_input = cv.normalize(segment_input, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        segment_input = cv.resize(segment_input, img_segments[indMarkers].shape, interpolation=cv.INTER_AREA)
        # img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        """
        filename = "marker_" + str(indMarkers) + ".png"
        matplotlib.image.imsave(filename, img, cmap='gray')
        """

        (contours_, _) = cv.findContours(segment_input, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        areas_ = []
        poly_approx_ = []
        for i, contour in enumerate(contours_):
            areas_.append(cv.contourArea(contour))
            perimeter = cv.arcLength(contour, True)
            poly_approx_.append(cv.approxPolyDP(contours_[i], perimeter * 0.001, True))
        assert len(areas_) > 0, (400, 'Missing marker(s) from the camera image')
        sorted_area_index = np.argsort(areas_)
        poly_ = np.squeeze(poly_approx_[sorted_area_index[-1]])
        if poly_.shape[0] > 2:
            centre_, radius_ = polylabel([poly_.tolist()], with_distance=True)
            c_local = [int(centre_[1]), int(centre_[0])]
            c_global = [c_local[0] + segment_origin[indMarkers][1],
                        c_local[1] + segment_origin[indMarkers][0]]
            centers_local.append(c_local)
            centers_global.append(c_global)
    """
    if GENERATE_IMAGE is True:
        for idx in range(len(img_segments)):
            cv.circle(img_segments[idx], (int(centers_local[idx][0]), int(centers_local[idx][1])), 10, (255), -1)
            plt.subplot(2, 5, idx + 1), plt.imshow(img_segments[idx])
            plt.title('{}:{}'.format(idx + 1, img_segments[idx].shape))
            plt.xticks([]), plt.yticks([])
        plt.show()
    """
    # visualize
    """
    if GENERATE_IMAGE is True:
        img_plot = img_input.copy()
        for i in range(10):
            cv.drawMarker(img_plot, (centers_global[i][0], centers_global[i][1]), (255, 0, 0), thickness=2)
        visualize(img_plot, title="Marked full image")
    """

    marker_corners = update_markers(centers_global)
    marker_radius = 0  # 2018 - 1955
    return marker_corners, marker_radius


def find_area(storage_id_str, img_input, return_image=False):
    # variables
    # visualize(img_input)
    generate_image = SHOW_PLOTS or return_image

    # storage_id = int(storage_id_str) - 1
    storage_id = 4 - int(storage_id_str)
    kernel_dilation = np.ones((10, 10), np.uint16)  # dilation kernel_dilation

    img_input = resize_to_predefined(img_input)
    img_gray = cv.cvtColor(img_input, cv.COLOR_BGR2GRAY)

    # get marker locations
    markers, delta = get_markers(img_gray)

    if generate_image is True:
        img_plot = img_input.copy()
        for i in range(4):
            cv.drawMarker(img_plot, (markers[storage_id][i][0], markers[storage_id][i][1]), (0, 0, 255), thickness=4)
        visualize(img_plot, title="Markers of selected storage")
        for j in range(4):
            img_plot = img_input.copy()
            for i in range(4):
                cv.drawMarker(img_plot, (markers[j][i][0], markers[j][i][1]), (0, 0, 255), thickness=4)
            visualize(img_plot, title="Main code" + str(j))

    # select the correct storage area
    # Here, specify the input coordinates for corners in the order of TL, TR, BR, BL
    marker_rectangle = np.float32([markers[storage_id][0], markers[storage_id][1],
                                   markers[storage_id][2], markers[storage_id][3]])
    # print(marker_rectangle)
    # delta = 0  # += 20

    # treat camera distortions ---------------------------
    # read input width and height
    height_img, width_img = img_gray.shape[:2]
    # get top and left dimensions and set to corner_coordinates dimensions of target rectangle
    width = round(math.hypot(marker_rectangle[0, 0] - marker_rectangle[1, 0],
                             marker_rectangle[0, 1] - marker_rectangle[1, 1]))
    height = round(math.hypot(marker_rectangle[0, 0] - marker_rectangle[3, 0],
                              marker_rectangle[0, 1] - marker_rectangle[3, 1]))
    # top left coordinates
    x_top_left, y_top_left = marker_rectangle[0, 0], marker_rectangle[0, 1]
    # corner_coordinates coordinates of corners in order TL, TR, BR, BL
    corner_coordinates = np.float32([
        [x_top_left, y_top_left], [x_top_left + width - 1, y_top_left],
        [x_top_left + width - 1, y_top_left + height - 1], [x_top_left, y_top_left + height - 1]
    ])
    # print(corner_coordinates)

    # compute perspective matrix
    perspective_matrix = cv.getPerspectiveTransform(marker_rectangle, corner_coordinates)
    # Perspective transformation setting and corner_coordinates size is the same as the input image size ???
    img_straighten = cv.warpPerspective(img_gray, perspective_matrix, (width_img, height_img), cv.INTER_LINEAR,
                                        borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    """
    if generate_image is True:
        img_straighten_color = cv.warpPerspective(img_input, perspective_matrix, (width_img, height_img),
                                                  cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # visualize(img_straighten_color, title="Distortion correction")
    """

    new_markers = get_translated_points(markers[storage_id], perspective_matrix)

    # filtering
    img_blurred = cv.bilateralFilter(img_straighten, 8, 25, 25)
    img_blurred_final = cv.GaussianBlur(img_blurred, (3, 3), 0)

    # fix brightness
    bright_sigma = 0.33
    image_median = np.median(img_blurred_final)
    if image_median > 191:  # light images
        brightness_tuner = 2
    elif image_median > 127:
        brightness_tuner = 1
    elif image_median < 63:  # dark
        brightness_tuner = 2
    else:
        brightness_tuner = 1
    brightness_lower = int(max(0, (1.0 - brightness_tuner * bright_sigma) * image_median))
    brightness_upper = int(min(255, (1.0 + brightness_tuner * bright_sigma) * image_median))

    # find edges
    edges = cv.Canny(img_blurred_final, brightness_lower, brightness_upper)
    img_dilated = cv.dilate(edges, kernel_dilation, iterations=1)

    # visualize(edges, title="Edge detection")
    """
    img_cropped = img_dilated[
                  :markers[storage_id][3][1],
                  markers[storage_id][3][0] + delta:
                  markers[storage_id][2][0] - delta
                  ]
    """
    img_cropped = img_dilated[
                  :new_markers[3][1],
                  new_markers[3][0]:new_markers[2][0]
                  ]

    # making a square from markers
    width_cropped = np.minimum(new_markers[1][0] - new_markers[0][0], new_markers[2][0] - new_markers[3][0])
    print(width_cropped)
    # Calculate scaling from pixels to actual distances
    scale_factor = MARKER_GAP / width_cropped  # need to fix, 280 mm
    height_markers = np.minimum(new_markers[3][1] - new_markers[0][1], new_markers[2][1] - new_markers[1][1])
    height_cropped = int(round(img_cropped.shape[0] * width_cropped / height_markers))
    img_cropped = cv.resize(img_cropped, (width_cropped, height_cropped), interpolation=cv.INTER_AREA)
    markers_cropped = [
        [0, height_cropped - width_cropped - 1], [width_cropped - 1, height_cropped - width_cropped - 1],
        [0, height_cropped - 1], [width_cropped - 1, height_cropped - 1]
    ]

    img_morphed = cv.morphologyEx(img_cropped, cv.MORPH_CLOSE, kernel_dilation)
    # get contours
    (contours_all, _) = cv.findContours(img_morphed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = {}
    contour_areas = []
    polygon_approx = []

    for index, contour in enumerate(contours_all):
        contour_areas.append(cv.contourArea(contour))
        perimeter = cv.arcLength(contour, True)
        # approximate to a polygon with given epsilon value
        polygon_approx.append(
            cv.approxPolyDP(contours_all[index], perimeter * 0.001, True))
        contours[index] = {
            "contour": contour,
            "area": contour_areas[index],
            "perimeter": perimeter,
            "polygon_approx": polygon_approx[index],
            "approxLen": len(polygon_approx)
        }

    # sorting area indexes
    sorted_indices_area = np.argsort(contour_areas)
    # print(len(polygon_approx[sorted_indices_area[-1]]), len(polygon_approx[sorted_indices_area[-2]]))

    # detecting weather there is inside edges

    # calculating radius and center
    poly = np.squeeze(polygon_approx[sorted_indices_area[-2]])  # hardcoded to be 2nd largest
    centre = [0., 0.]
    radius = 0.
    if poly.shape[0] > 2:
        centre, radius = polylabel([poly.tolist()], with_distance=True)

    # translate to the original coordinates
    # x_mm = storage_id * MARKER_GAP + pixel_to_mm(centre[0], ratio=scale_factor)
    # y_mm = pixel_to_mm(centre[1] - markers_cropped[0][1], ratio=scale_factor)
    print(centre[0], markers_cropped[3][1] - centre[1])
    x_mm = 4 * MARKER_GAP - (storage_id * MARKER_GAP + pixel_to_mm(centre[0], ratio=scale_factor))
    y_mm = pixel_to_mm(markers_cropped[3][1] - centre[1], ratio=scale_factor)
    radius_mm = pixel_to_mm(radius, ratio=scale_factor)
    confidence = compute_confidence(MARKER_GAP / 2, MARKER_GAP + TABLE_OFFSET,
                                    pixel_to_mm(centre[0], ratio=scale_factor), y_mm, radius_mm)

    """"""
    # extracted contour, visualization only
    if generate_image is True:
        img_plot = img_input.copy()
        for i in range(4):
            cv.drawMarker(img_plot, (markers[storage_id][i][0], markers[storage_id][i][1]), (0, 0, 255), thickness=4)
        visualize(img_plot, title="Markers of selected storage")

        img_straighten_color = cv.warpPerspective(img_input, perspective_matrix, (width_img, height_img),
                                                  cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
        visualize(img_straighten_color, title="Distortion correction")
        visualize(edges, title="Edge detection")

        image_visual = img_straighten_color[
                       :new_markers[3][1],
                       new_markers[3][0]:new_markers[2][0]
                       ]
        image_visual = cv.resize(image_visual, (width_cropped, height_cropped), interpolation=cv.INTER_AREA)
        for i in range(4):
            cv.drawMarker(image_visual, (markers_cropped[i][0], markers_cropped[i][1]), (0, 0, 255), thickness=4)
        cv.drawMarker(image_visual, (int(centre[0]), int(centre[1])), (0, 0, 255), thickness=4)
        cv.circle(image_visual, (int(centre[0]), int(centre[1])), int(radius), (34, 139, 34), -1)
        visualize(image_visual, title="Area")
    """"""
    if generate_image and return_image is True:
        return image_visual
    else:
        return x_mm, y_mm, radius_mm, confidence


def resize_to_predefined(img_input):
    width = 4056
    height = 3040
    dim = (width, height)

    # resize image
    return cv.resize(img_input, dim, interpolation=cv.INTER_AREA)


def compute_confidence(mark_center_x, mark_center_y, center_x, center_y, radius_):
    # confidence for the radius
    radius_shift = [np.pi * np.maximum(np.minimum(radius_ / mark_center_x, 1.), 0)]
    confidence_radius = smooth_wave_filter(radius_shift, steepness=.3)

    # confidence for the center
    max_radius_x = mark_center_x - radius_
    shift_x = np.minimum(np.absolute(center_x - mark_center_x) / (2 * max_radius_x) + .5, 1.)
    # confidence_center_x = smooth_wave_filter([np.pi * shift_x], steepness=.1)

    new_mark_center_y = (mark_center_y + radius_) / 2
    max_radius_y = new_mark_center_y - radius_
    shift_y = np.minimum(np.absolute(center_y - new_mark_center_y) / (2 * max_radius_y) + .5, 1.)
    # confidence_center_y = smooth_wave_filter([np.pi * shift_y], steepness=.1)

    # return confidence_radius * confidence_center_x * confidence_center_y * 100
    confidence_center = smooth_wave_filter([np.pi * np.sqrt((shift_x ** 2 + shift_y ** 2) / 2)], steepness=.1)
    # print(confidence_radius, confidence_center)
    return confidence_radius * confidence_center * 100


def smooth_wave_filter(input_value, steepness=0.1):
    deg_1 = 1.
    deg_2 = 1.
    for _ in input_value:
        deg_1 = deg_1 * np.sin(_)
        deg_2 = deg_2 * np.sin(_) * np.sin(_)
    return deg_1 * np.sqrt(1 + steepness) / np.sqrt(deg_2 + steepness)


if __name__ == '__main__':  # avoid import and run ensure file is run directly in python
    # image read
    img_input = cv.imread('capture.jpg')  # update with camera input

    for _ in range(3, 4):
        storage_id = str(_ + 1)
        camera_hostname = ""
        camera_id = ""

        x_mm, y_mm, radius_mm, confidence = find_area(storage_id, img_input)
        print("-------------------")
        print("xcoordinate", x_mm,
              "ycoordinate", y_mm,
              "radius-mm", radius_mm,
              "confidence", confidence,
              )
