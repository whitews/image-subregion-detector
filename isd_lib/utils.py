import cv2
import numpy as np

# Define color ranges in HSV with lower & upper ranges
# NOTE: HSV value range in OpenCV:
#   H: 0 -> 180
#   S: 0 -> 255
#   V: 0 -> 255
#
# NOTE: In HSV, red wraps around the beginning and end of the hue range
#
# Major colors encompass a range of ~40 H values
# Minor colors encompass a range of ~20 H values
HSV_RANGES = {
    # red is a major color
    'red': [
        {
            'lower': np.array([0, 39, 64]),
            'upper': np.array([20, 255, 255])
        },
        {
            'lower': np.array([161, 39, 64]),
            'upper': np.array([180, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([21, 39, 64]),
            'upper': np.array([40, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([41, 39, 64]),
            'upper': np.array([80, 255, 255])
        }
    ],
    # cyan is a minor color
    'cyan': [
        {
            'lower': np.array([81, 39, 64]),
            'upper': np.array([100, 255, 255])
        }
    ],
    # blue is a major color
    'blue': [
        {
            'lower': np.array([101, 39, 64]),
            'upper': np.array([140, 255, 255])
        }
    ],
    # violet is a minor color
    'violet': [
        {
            'lower': np.array([141, 39, 64]),
            'upper': np.array([160, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 10% of V
    'black': [
        {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 255, 63])
        }
    ],
    # gray is all H values, lower 15% of S, & between 11-89% of V
    'gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([180, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'white': [
        {
            'lower': np.array([0, 0, 229]),
            'upper': np.array([180, 38, 255])
        }
    ]
}


def find_regions(
        src_img,
        target_img,
        bg_colors=None,
        dilate=2,
        min_area=0.5,
        max_area=2.0
):
    """
    Finds regions in source image that are similar to the target image.

    Args:
        src_img: 3-D NumPy array of pixels in HSV (source image)
        target_img: 3-D NumPy array of pixels in HSV (target image)
        bg_colors: list of color names to use for background colors, if
            None the dominant color in the source image will be used
        dilate: # of dilation iterations performed on masked image
        min_area: minimum area cutoff percentage (compared to target image)
            for returning matching sub-regions
        max_area: maximum area cutoff percentage (compared to target image)
            for returning matching sub-regions

    Returns:
        Sub-region mask as 2-D NumPy array (unsigned 8-bit integers) with the
        same width and height as the source image. Matching sub-regions have
        pixel values of 255 and non-matching pixels are 0.

    Raises:
        tbd
    """

    # if no bg colors are specified, determine dominant color range
    # for the 'background' in the source image
    if bg_colors is None:
        bg_colors = [find_dominant_color(src_img)]

    # determine # of pixels of each color range found in the target
    target_color_profile = get_color_profile(target_img)

    # find common color ranges in target (excluding the bg_colors)
    feature_colors = get_common_colors(target_color_profile, bg_colors)

    # create masks from feature colors
    mask = create_mask(src_img, feature_colors)
    target_mask = create_mask(target_img, feature_colors)

    # dilate masks
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilate)
    target_mask = cv2.dilate(target_mask, kernel, iterations=dilate)

    # fill holes in mask using contours
    mask = fill_holes(mask)
    target_mask = fill_holes(target_mask)

    # determine target mask area
    feature_area = np.sum(target_mask) / 255

    # remove contours below min_area and above max_area
    min_pixels = int(feature_area * min_area)
    max_pixels = int(feature_area * max_area)
    mask, rectangles = filter_blobs_by_size(mask, min_pixels, max_pixels)

    # return contours & bounding rectangle coordinates
    return mask, rectangles


def find_dominant_color(hsv_img):
    """
    Finds dominant color in given HSV image array

    Args:
        hsv_img: HSV pixel data (3-D NumPy array)

    Returns:
        Text string for dominant color range (from HSV_RANGES keys)

    Raises:
        tbd
    """
    color_profile = get_color_profile(hsv_img)
    dominant_color = max(color_profile, key=lambda k: color_profile[k])

    return dominant_color


def get_color_profile(hsv_img):
    """
    Finds color profile as pixel counts for color ranges in HSV_RANGES

    Args:
        hsv_img: HSV pixel data (3-D NumPy array)

    Returns:
        Text string for dominant color range (from HSV_RANGES keys)

    Raises:
        tbd
    """
    h, s, v = get_hsv(hsv_img)

    color_profile = {}

    for color, color_ranges in HSV_RANGES.iteritems():
        color_profile[color] = 0

        for color_range in color_ranges:
            pixel_count = np.sum(
                np.logical_and(
                    h >= color_range['lower'][0],
                    h <= color_range['upper'][0]
                ) &
                np.logical_and(
                    s >= color_range['lower'][1],
                    s <= color_range['upper'][1]
                ) &
                np.logical_and(
                    v >= color_range['lower'][2],
                    v <= color_range['upper'][2]
                )
            )

            color_profile[color] += pixel_count

    return color_profile


def get_hsv(hsv_img):
    """
    Returns flattened hue, saturation, and values from given HSV image.
    """
    hue = hsv_img[:, :, 0].flatten()
    sat = hsv_img[:, :, 1].flatten()
    val = hsv_img[:, :, 2].flatten()

    return hue, sat, val


def get_common_colors(color_profile, bg_colors, prevalence=0.1):
    """
    Finds colors in a color profile (excluding bg colors) that exceed prevalence
    """
    total = sum(color_profile.values())
    for bg_color in bg_colors:
        total -= color_profile[bg_color]
    common_colors = []

    for color, count in color_profile.iteritems():
        if color in bg_colors:
            continue

        if count > prevalence * total:
            common_colors.append(color)

    return common_colors


def create_mask(hsv_img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask


def fill_holes(mask):
    """
    Fills holes in a given binary mask.
    """
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    new_mask, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        cv2.drawContours(new_mask, [cnt], 0, 255, -1)

    return new_mask


def filter_blobs_by_size(mask, min_pixels, max_pixels):
    """
    Filters a given binary mask keeping blobs within a min & max size
    """
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    new_mask, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = []

    for c in contours:
        c_area = cv2.contourArea(c)
        if min_pixels <= c_area <= max_pixels:
            cv2.drawContours(new_mask, [c], 0, 255, -1)
            rectangles.append(cv2.boundingRect(c))
        else:
            cv2.drawContours(new_mask, [c], 0, 0, -1)

    return new_mask, rectangles
