import colorsys
import hashlib
from functools import lru_cache

_COLORS = {
    # 绿色
    'green': (0, 255, 0),
    # 红色
    'red': (0, 0, 255),
    # 蓝色
    'blue': (255, 0, 0),
    # 黄色
    'yellow': (0, 255, 255),
    # 紫色
    'purple': (255, 0, 255),
    # 青色
    'cyan': (255, 255, 0),
    # 黑色
    'black': (0, 0, 0),
    # 白色
    'white': (255, 255, 255),
    # 灰色
    'gray': (128, 128, 128),
    # 深灰色
    "GhostWhite": (248, 248, 255)
}


def get_luminance(bgr: tuple[int, int, int]) -> float:
    # BGR to RGB
    b, g, r = bgr
    return 0.299 * r + 0.587 * g + 0.114 * b  # ITU-R BT.601


def ensure_contrast(text_color: tuple[int, int, int], background_color: tuple[int, int, int]) -> tuple[int, int, int]:
    text_lum = get_luminance(text_color)
    bg_lum = get_luminance(background_color)

    contrast = abs(text_lum - bg_lum)
    if contrast < 128:  # 亮度差太小，自动调深背景
        if text_lum > 128:
            return (32, 32, 32)  # 改成深灰色背景
        else:
            return (255, 255, 255)  # 改成白色背景
    return background_color


def get_vibrant_color_from_key(key: str) -> tuple[int, int, int]:
    # 使用 MD5 哈希获取稳定值
    md5_bytes = hashlib.md5(key.encode()).digest()  # 16 字节
    hash_bytes = md5_bytes[-3:]  # 从中随机选 3 个字节

    # 使用前 3 字节作为 HSV 分量来源
    h = hash_bytes[0] / 255.0  # Hue: [0, 1]
    s = 0.8 + (hash_bytes[1] % 50) / 100.0  # Saturation: [0.8, 1.3] (上限1.0后裁剪)
    v = 0.8 + (hash_bytes[2] % 50) / 100.0  # Value: [0.8, 1.3] (上限1.0后裁剪)

    s = min(s, 1.0)
    v = min(v, 1.0)

    # 转为 RGB，再转为 BGR（OpenCV格式）
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)  # OpenCV 使用 BGR


@lru_cache(maxsize=1000)
def getcolor(key="red", bgr=True):
    """
    Get a color by key, either as BGR tuple or hex string.

    这是一个稳定的方法，相同的 key 总是返回相同的颜色。
    :param key:
    :param bgr:
    :return:
    """
    key = str(key)

    if key in _COLORS:
        cl_bgr = _COLORS[key]
    else:
        b, g, r = get_vibrant_color_from_key(key)
        cl_bgr = (b, g, r)
        _COLORS[key] = cl_bgr
    return cl_bgr if bgr else f'#{r:02x}{g:02x}{b:02x}'
