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



def negotiate_layout(candidates, max_error_deg, max_bytes):
    """在候选布局中挑选：返回 dict(selected=索引或 None, rejected=[(索引, 理由码)]。

    理由码至少区分误差超限与带宽超限。
    """
    raise NotImplementedError("negotiate_layout 尚未实现")


def pack_pixel_half(normal, rough, mat):
    """半精度法线版本：返回整数位串，其余通道语义不变。"""
    raise NotImplementedError("pack_pixel_half 尚未实现")


def decode_half(packed):
    """解半精度版本：返回 dict(normal=(x,y,z), rough=float, mat=int)。"""
    raise NotImplementedError("decode_half 尚未实现")


def decode_aligned(packed, layout):
    """按布局解码：返回 dict(normal, rough, mat)；位宽不一致必须报错。"""
    raise NotImplementedError("decode_aligned 尚未实现")


def bandwidth_cost(layout, width, height):
    """返回该布局在给定分辨率下的字节数。"""
    raise NotImplementedError("bandwidth_cost 尚未实现")
