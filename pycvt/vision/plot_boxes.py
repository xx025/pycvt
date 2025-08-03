from typing import Union

import cv2
import numpy
import numpy as np
from distinctipy.distinctipy import get_rgb256, get_text_color
from easyfont import getfont

from pycvt.clolors.colors import getcolor
from pycvt.vision.utils import render_text_image


def draw_bounding_boxes(
        image: numpy.ndarray,
        boxes: Union[np.ndarray, list],
        labels: list = None,
        colors=None,
        line_width=None,
        font=None,
        font_size=None
):
    h, w = image.shape[:2]
    boxes = np.asarray(boxes, dtype=int)
    n = len(boxes)
    line_width = line_width if line_width else max(int(0.003 * min(w, h)), 2)
    font_size = font_size if font_size else line_width * 6
    font = font if font else getfont()

    if colors is None:
        if labels:
            colors = np.array([getcolor(label) for label in labels])
        else:
            colors = np.array([getcolor()] * n)
    labels = labels if labels else [None] * n
    boxes_colors = np.asarray(colors, dtype=int)

    for box, label, color in zip(boxes, labels, boxes_colors):
        xmin, ymin, xmax, ymax = box
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color.tolist(), line_width, lineType=cv2.LINE_AA)
        if label:
            text_color = get_rgb256(get_text_color(color / 255))
            text_rendered = render_text_image(label, font, font_size, text_color=text_color, bg_color=color)[..., :3]
            text_h, text_w = text_rendered.shape[:2]
            try_poses = [(xmin, ymax + line_width + 2), (xmin, ymin - text_h - line_width), (xmin + line_width + 2, ymin + line_width + 2)]
            for xstart, ystart in try_poses:
                if xstart < 0 or ystart < 0 or xstart + text_w > w or ystart + text_h > h:
                    continue
                image[ystart:ystart + text_h, xstart:xstart + text_w] = text_rendered
                break
    return image


if __name__ == "__main__":
    from PIL import Image

    img = np.ones((800, 800, 3), dtype=np.uint8) * 255  # 创建一个黑色背景图像
    boxes = [[50, 50, 800, 800], [300, 100, 550, 300], [5, 5, 900, 900]]
    labels = ["Object A", "Object B 你好，世界！", "MML"]  # 标签列表
    img_with_boxes = draw_bounding_boxes(img, boxes, labels, line_width=2)
    img_with_boxes = Image.fromarray(img_with_boxes)
    img_with_boxes.show()  # 显示图像
