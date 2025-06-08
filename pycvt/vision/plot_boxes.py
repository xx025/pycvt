from copy import deepcopy

import cv2
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from easyfont import getfont

from pycvt.clolors.colors import getcolor, ensure_contrast


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

    text_color = text_color or getcolor(text)
    background_color = background_color or getcolor("GhostWhite")
    background_color = ensure_contrast(text_color, background_color)  # 确保文本颜色和背景颜色有足够对比度

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img_pil)

    spacing = font_size // 3
    half_spacing = spacing // 2

    text_bbox = draw.textbbox((0, 0), text, font=font, font_size=font_size, spacing=spacing)
    text_bbox_h = text_bbox[3]
    text_bbox_w = text_bbox[2]

    bgpos = (
        position[0] - text_bbox_w - half_spacing,
        position[1] + spacing,
        position[0] + spacing,
        position[1] + text_bbox_h + spacing
    )

    text_pos = position[0] - text_bbox_w, position[1] + half_spacing
    draw.rectangle(bgpos, fill=background_color)
    draw.text(text_pos, text, font=font, fill=text_color, spacing=spacing)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_bounding_boxes(
        image: numpy.ndarray,
        boxes: list,
        labels=None,
        colors=None,
        width=None,
        font=None,
        font_size=None
):
    img_copy = deepcopy(image)

    count = len(boxes)
    w, h = img_copy.shape[1], img_copy.shape[0]

    if colors is None:
        colors = list(map(getcolor, labels)) if labels else [getcolor()] * count

    line_width = width if width else max(int(0.0035 * min(w, h)), 2)
    font_size = font_size if font_size else max(int(0.018 * min(w, h)), 2)

    font = font if font else getfont()
    for idx, box in enumerate(boxes):
        color = colors[idx]
        xmin, ymin, xmax, ymax = box
        img_copy = cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, line_width)
        if labels:
            label = labels[idx]
            img_copy = draw_text(
                img_copy,
                label,
                (xmax, ymax),
                text_color=color[::-1],  # OpenCV uses BGR, PIL uses RGB
                font_path=font,
                font_size=font_size
            )
    return img_copy
