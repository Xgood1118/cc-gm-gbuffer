# -*- coding: utf-8 -*-
"""G-Buffer 打包与精度"""

from dataclasses import dataclass


@dataclass
class GBuffer:
    """一个像素的 G-Buffer 数据。"""
    normal: tuple
    rough: float
    mat: int


def GBuffer:
    """class"""
    raise NotImplementedError("GBuffer 尚未实现")


def pack_pixel(normal, rough, mat):
    """打包成一个整数"""
    raise NotImplementedError("pack_pixel 尚未实现")


def unpack_pixel(packed):
    """解包"""
    raise NotImplementedError("unpack_pixel 尚未实现")


def measure_error(orig, back):
    """算角度与粗糙度误差"""
    raise NotImplementedError("measure_error 尚未实现")

