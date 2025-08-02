from typing import Union

import cv2
import numpy
import numpy as np
from distinctipy.distinctipy import get_rgb256, get_text_color
from easyfont import getfont

from pycvt.clolors.colors import getcolor
from pycvt.vision.utils import render_text_image


def draw_text(
        img,
        text,
        position,
        font_path=None,
        font_size=None,
        text_color=None,
        background_color=None
):
    """
    在图像上绘制文本
    :param img:  图像对象（OpenCV格式）
    :param text:  要绘制的文本(受字体支持的字符)
    :param position:  文本位置，建议使用 xmax, ymax 作为位置 ，防止覆盖包围框
    :param font_path: 字体路径
    :param font_size:  字体大小
    :param text_color:  文本颜色
    :param background_color:  文本背景颜色
    :return:
    """
    # 将 OpenCV 图像转换为 PIL 图像

    if font_path is None:  # 若未指定字体路径，则使用默认字体
        font_path = getfont()

    h, w = img.shape[:2]
    font_size = font_size if font_size else max(int(0.003 * min(w, h)), 2)
    background_color = background_color if background_color is not None else getcolor()
    if text_color is None:
        text_color = get_rgb256(get_text_color(background_color / 255))  # 获取文本颜色，确保对比度

    text_rendered = render_text_image(
        text,
        font_path,
        font_size,
        text_color=text_color,
        bg_color=background_color
    )
    text_rendered = text_rendered[..., :img.shape[2]]
    text_h, text_w = text_rendered.shape[:2]
    x, y = position
    x = max(0, x)
    y = max(0, y)
    x_end = min(w, x + text_w)
    y_end = min(h, y + text_h)
    img[y:y_end, x:x_end] = text_rendered[:y_end - y, :x_end - x]
    return img


def draw_bounding_boxes(
        image: numpy.ndarray,
        boxes: Union[np.ndarray, list],
        labels: list = None,
        colors=None,
        width=None,
        font=None,
        font_size=None
):
    h, w = image.shape[:2]
    boxes = np.asarray(boxes, dtype=int)
    n = len(boxes)
    line_width = width if width else max(int(0.003 * min(w, h)), 2)
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
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color.tolist(), line_width)
        if label:
            text_color = get_rgb256(get_text_color(color / 255))  # 获取文本颜色，确保对比度
            text_rendered = render_text_image(
                label,
                font,
                font_size,
                text_color=text_color,
                bg_color=color
            )[..., :3]  # 确保只取 RGB 通道
            text_h, text_w = text_rendered.shape[:2]
            x, y = xmin, ymax
            x = max(0, x)
            y = max(0, y)
            x_end = min(w, x + text_w)
            y_end = min(h, y + text_h)
            image[y:y_end, x:x_end] = text_rendered[:y_end - y, :x_end - x]
    return image


if __name__ == "__main__":
    from PIL import Image
    img = np.ones((800, 800, 3), dtype=np.uint8) * 255  # 创建一个黑色背景图像
    boxes = [[50, 50, 200, 200], [300, 100, 550, 300]]
    labels = ["Object A", "Object B 你好，世界！"]  # 标签列表
    img_with_boxes = draw_bounding_boxes(img, boxes, labels, width=3)
    img_with_boxes = Image.fromarray(img_with_boxes)
    img_with_boxes.show()  # 显示图像
