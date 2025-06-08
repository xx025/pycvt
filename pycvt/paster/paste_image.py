from copy import deepcopy

import numpy as np

from pycvt.utils.image_utils import convert_rgba, blend_rgba


def paste_image(
        bg_img,
        fg_img,
        position=None,
        alpha=1.0,
):
    """
    fg_img will be pasted onto bg_img at the specified position with the given alpha blending factor.

    1. If position is None, a random position will be chosen within the bounds of the background image.
    2. IF position + fg_img exceeds the bounds of bg_img, it will be clipped to fit within bg_img.

    :param bg_img: Background image, can be a numpy array or a PIL image.
    :param fg_img: Foreground image, can be a numpy array or a PIL image.
    :param position: Tuple (y, x) indicating where to paste the foreground image on the background image.
    :param alpha: Blending factor for the foreground image, where 0.0 is fully transparent and 1.0 is fully opaque.
    :return: Tuple containing the modified background image and the bounding box of the pasted foreground image.
    """

    bg_img = deepcopy(bg_img)
    fg_img = deepcopy(fg_img)

    bg_img = convert_rgba(bg_img)
    fg_img = convert_rgba(fg_img)

    img_h, img_w, = bg_img.shape[:2]
    fg_h, fg_w, = fg_img.shape[:2]

    alpha = np.clip(alpha, 0.0, 1.0)

    if position is None:
        position = np.random.randint(0, img_h - fg_h // 2), np.random.randint(0, img_w - fg_w // 2),

    ymin, xmin = position
    ymax, xmax = min(img_h, position[0] + fg_h), min(img_w, position[1] + fg_w)


    paste_size= (xmax - xmin, ymax - ymin)

    bg_crop = bg_img[ymin:ymax, xmin:xmax]
    fg_crop = fg_img[:paste_size[1], :paste_size[0]]

    blended = blend_rgba(bg_crop, fg_crop, fgweight=alpha)

    bg_img[ymin:ymax, xmin:xmax] = blended

    return bg_img, (xmin, ymin, xmax, ymax)
