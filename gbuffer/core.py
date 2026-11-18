# -*- coding: utf-8 -*-
"""G-Buffer 打包与精度。

布局约定
========

32 位整包（``pack_pixel`` / ``unpack_pixel``）：

- bits [15:0]  法线：八面体（octahedral）编码，u/v 各 8 位
- bits [23:16] 粗糙度：8 位，[0,1] 均匀量化
- bits [31:24] 材质编号：8 位，整数原样存放，精确往返

半精度整包（``pack_pixel_half`` / ``decode_half``），共 64 位：

- bits [47:0]  法线三个分量，各 16 位半精度浮点（IEEE 754 binary16）
- bits [55:48] 粗糙度：8 位
- bits [63:56] 材质编号：8 位

位宽不足时的分配优先级（``allocate_bits``）：材质编号优先（必须精确往返），
剩余位全部给粗糙度。

误差上界（推导见 README）：

- 八面体编码每分量 n 位（2**n - 2 个量化级，0 精确可表示）时，
  法线角度误差上界为 ``2*asin(3 / (sqrt(2) * (2**n - 2)))``；
  32 位整包 n=8，上界约 0.958 度。
- 半精度法线分量舍入误差 <= 2**-11，角度误差上界约 0.097 度，
  严格小于 32 位整包上界。
- 粗糙度 b 位时误差上界为 ``0.5 / (2**b - 1)``。
- 材质编号不量化，误差恒为 0。

边界行为（确定性）：

- 零长度（或非有限）法线：按 (0, 0, 1) 处理。
- 粗糙度越界：截断到 [0, 1]。
- 材质编号越界（负或 >= 2**material_bits）：抛 ValueError。
"""

import math
import struct
from dataclasses import dataclass

__all__ = [
    "GBuffer",
    "pack_pixel",
    "unpack_pixel",
    "measure_error",
    "negotiate_layout",
    "pack_pixel_half",
    "decode_half",
    "decode_aligned",
    "pack_aligned",
    "bandwidth_cost",
    "allocate_bits",
    "normal_error_bound_deg",
    "rough_error_bound",
    "NORMAL_BITS",
    "ROUGH_BITS",
    "MAT_BITS",
    "TOTAL_BITS",
    "TAG_BITS",
    "HALF_TOTAL_BITS",
    "HALF_NORMAL_ERROR_BOUND_DEG",
    "REASON_ERROR",
    "REASON_BANDWIDTH",
]

# ---- 32 位整包布局 ----
NORMAL_BITS = 16
ROUGH_BITS = 8
MAT_BITS = 8
TOTAL_BITS = NORMAL_BITS + ROUGH_BITS + MAT_BITS

# ---- 半精度整包布局 ----
HALF_NORMAL_BITS = 48  # 3 x 16 bit binary16
HALF_TOTAL_BITS = HALF_NORMAL_BITS + ROUGH_BITS + MAT_BITS  # 64

# ---- decode_aligned 布局标签位宽 ----
TAG_BITS = 8

# ---- 布局协商淘汰理由码 ----
REASON_ERROR = "E_ERROR"        # 法线角度误差上界超限
REASON_BANDWIDTH = "E_BANDWIDTH"  # 总带宽超限

# 半精度法线角度误差上界（度）：2*asin(sqrt(3) * 2**-11)
HALF_NORMAL_ERROR_BOUND_DEG = math.degrees(
    2.0 * math.asin(math.sqrt(3.0) * 2.0 ** -11)
)


@dataclass
class GBuffer:
    """一个像素的 G-Buffer 数据。"""
    normal: tuple
    rough: float
    mat: int


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _safe_normalize(normal):
    """归一化；零长度或非有限输入确定性地返回 (0, 0, 1)。"""
    nx, ny, nz = (float(c) for c in normal)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0 or not math.isfinite(length):
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def _oct_encode(nx, ny, nz):
    """单位法线 -> 八面体坐标 (u, v) ∈ [-1, 1]^2。"""
    s = abs(nx) + abs(ny) + abs(nz)
    x, y = nx / s, ny / s
    if nz < 0.0:
        x, y = ((1.0 - abs(y)) * (1.0 if x >= 0.0 else -1.0),
                (1.0 - abs(x)) * (1.0 if y >= 0.0 else -1.0))
    return x, y


def _oct_decode(u, v):
    """八面体坐标 -> 重新归一化的单位法线。"""
    z = 1.0 - abs(u) - abs(v)
    x, y = u, v
    if z < 0.0:
        x, y = ((1.0 - abs(y)) * (1.0 if x >= 0.0 else -1.0),
                (1.0 - abs(x)) * (1.0 if y >= 0.0 else -1.0))
    return _safe_normalize((x, y, z))


def _quant_unit(value, bits):
    """[0,1] 均匀量化到 [0, 2**bits - 1]；越界先截断（确定性）。"""
    if bits <= 0:
        return 0
    v = min(1.0, max(0.0, float(value)))
    q = int(round(v * ((1 << bits) - 1)))
    return min((1 << bits) - 1, max(0, q))


def _dequant_unit(q, bits):
    if bits <= 0:
        return 0.0
    return q / float((1 << bits) - 1)


def _quant_signed(value, bits):
    """[-1,1] 均匀量化到 [0, 2**bits - 2]（奇数级，0 精确可表示）。"""
    if bits <= 1:
        return 0
    levels = (1 << bits) - 2
    v = min(1.0, max(-1.0, float(value)))
    return int(round((v + 1.0) * 0.5 * levels))


def _dequant_signed(q, bits):
    if bits <= 1:
        return 0.0
    levels = (1 << bits) - 2
    return q / float(levels) * 2.0 - 1.0


def _check_mat(mat, bits):
    """材质编号必须能精确往返：越界直接报错（确定性结果）。"""
    limit = 1 << bits
    if not isinstance(mat, int) or isinstance(mat, bool) or not (0 <= mat < limit):
        raise ValueError("材质编号越界: %r（当前布局允许 0..%d）" % (mat, limit - 1))
    return mat


def _split_normal_bits(normal_bits):
    """法线位宽拆成 u/v 两个分量，v 拿多出来的一位。"""
    nu = normal_bits // 2
    return nu, normal_bits - nu


# ---------------------------------------------------------------------------
# 通用负载打包/解包（各通道独立位域，互不污染）
# ---------------------------------------------------------------------------

def _pack_payload(normal, rough, mat, normal_bits, rough_bits, mat_bits):
    nu, nv = _split_normal_bits(normal_bits)
    if nu < 1 or nv < 1:
        raise ValueError("法线位宽不足：每个分量至少需要 1 位")
    nx, ny, nz = _safe_normalize(normal)
    u, v = _oct_encode(nx, ny, nz)
    nval = _quant_signed(u, nu) | (_quant_signed(v, nv) << nu)
    rval = _quant_unit(rough, rough_bits)
    mval = _check_mat(mat, mat_bits)
    return nval | (rval << normal_bits) | (mval << (normal_bits + rough_bits))


def _unpack_payload(payload, normal_bits, rough_bits, mat_bits):
    nu, nv = _split_normal_bits(normal_bits)
    nval = payload & ((1 << normal_bits) - 1)
    rval = (payload >> normal_bits) & ((1 << rough_bits) - 1) if rough_bits else 0
    mval = (payload >> (normal_bits + rough_bits)) & ((1 << mat_bits) - 1) if mat_bits else 0
    u = _dequant_signed(nval & ((1 << nu) - 1), nu)
    v = _dequant_signed(nval >> nu, nv)
    return {
        "normal": _oct_decode(u, v),
        "rough": _dequant_unit(rval, rough_bits),
        "mat": mval,
    }


# ---------------------------------------------------------------------------
# 32 位整包
# ---------------------------------------------------------------------------

def pack_pixel(normal, rough, mat):
    """打包成一个整数（32 位）。同一输入两次打包结果一致。"""
    return _pack_payload(normal, rough, mat, NORMAL_BITS, ROUGH_BITS, MAT_BITS)


def unpack_pixel(packed):
    """解包 32 位整包，返回 dict(normal, rough, mat)；法线已重新归一化。"""
    if not isinstance(packed, int) or isinstance(packed, bool) \
            or not (0 <= packed < (1 << TOTAL_BITS)):
        raise ValueError("打包值必须是 [0, 2**%d) 的整数" % TOTAL_BITS)
    return _unpack_payload(packed, NORMAL_BITS, ROUGH_BITS, MAT_BITS)


def measure_error(orig, back):
    """算角度与粗糙度误差：orig 为 GBuffer，back 为解包 dict。"""
    on = _safe_normalize(orig.normal)
    bn = _safe_normalize(back["normal"])
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(on, bn))))
    return {
        "angle_deg": math.degrees(math.acos(dot)),
        "rough_err": abs(float(orig.rough) - float(back["rough"])),
    }


# ---------------------------------------------------------------------------
# 误差上界
# ---------------------------------------------------------------------------

def normal_error_bound_deg(normal_bits):
    """八面体编码在指定位宽下的法线角度误差上界（度）。

    推导：u/v 每分量 n 位、2**n - 2 个量化级时，每分量量化误差
    eps = 1/(2**n - 2)。解码映射 h(u,v) = normalize(g(u,v)) 的
    Lipschitz 常数 <= 3（|Dg| <= sqrt(3)，折叠区 |g| 最小为 1/sqrt(3)，
    故 |D normalize| <= sqrt(3)），(u,v) 量化位移 <= sqrt(2)*eps，
    弦长误差 <= 3*sqrt(2)*eps，角度 <= 2*asin(3*eps/sqrt(2))。
    """
    nu, nv = _split_normal_bits(normal_bits)
    n = min(nu, nv)
    if n < 2:
        return 180.0
    eps = 1.0 / float((1 << n) - 2)
    return math.degrees(2.0 * math.asin(min(1.0, 3.0 * eps / math.sqrt(2.0))))


def rough_error_bound(rough_bits):
    """粗糙度量化误差上界。"""
    if rough_bits <= 0:
        return 1.0
    return 0.5 / float((1 << rough_bits) - 1)


# ---------------------------------------------------------------------------
# 位宽分配与布局协商
# ---------------------------------------------------------------------------

def allocate_bits(total_bits, rough_want, mat_want):
    """粗糙度与材质编号共用 total_bits 时的分配。

    明确优先级：材质编号优先（必须精确往返），剩余位全部给粗糙度。
    """
    total_bits = max(0, int(total_bits))
    mat_bits = max(0, min(int(mat_want), total_bits))
    rough_bits = max(0, min(int(rough_want), total_bits - mat_bits))
    return {"rough_bits": rough_bits, "material_bits": mat_bits}


def _layout_total_bits(layout):
    return int(layout["normal_bits"]) + int(layout["rough_bits"]) \
        + int(layout["material_bits"])


def negotiate_layout(candidates, max_error_deg, max_bytes):
    """在候选布局中挑选：返回 dict(selected=索引或 None, rejected=[(索引, 理由码)])。

    理由码：E_ERROR（法线角度误差上界超限）、E_BANDWIDTH（带宽超限）。
    可行布局中选误差上界最小者，并列时取带宽更小、索引更小者。
    """
    rejected = []
    feasible = []
    for i, c in enumerate(candidates):
        total = _layout_total_bits(c)
        nbytes = -(-total // 8)
        if nbytes > max_bytes:
            rejected.append((i, REASON_BANDWIDTH))
            continue
        bound = normal_error_bound_deg(int(c["normal_bits"]))
        if bound > max_error_deg:
            rejected.append((i, REASON_ERROR))
            continue
        feasible.append((bound, nbytes, i))
    selected = min(feasible)[2] if feasible else None
    return {"selected": selected, "rejected": rejected}


def bandwidth_cost(layout, width, height):
    """返回该布局在给定分辨率下的字节数。"""
    nbytes = -(-_layout_total_bits(layout) // 8)
    return int(width) * int(height) * nbytes


# ---------------------------------------------------------------------------
# 半精度法线整包（64 位）
# ---------------------------------------------------------------------------

def pack_pixel_half(normal, rough, mat):
    """半精度法线版本：返回整数位串，其余通道语义不变。"""
    nx, ny, nz = _safe_normalize(normal)
    hval = int.from_bytes(struct.pack("<eee", nx, ny, nz), "little")
    rval = _quant_unit(rough, ROUGH_BITS)
    mval = _check_mat(mat, MAT_BITS)
    return hval | (rval << HALF_NORMAL_BITS) | (mval << (HALF_NORMAL_BITS + ROUGH_BITS))


def decode_half(packed):
    """解半精度版本：返回 dict(normal=(x,y,z), rough=float, mat=int)。"""
    if not isinstance(packed, int) or isinstance(packed, bool) \
            or not (0 <= packed < (1 << HALF_TOTAL_BITS)):
        raise ValueError("半精度包必须是 [0, 2**%d) 的整数" % HALF_TOTAL_BITS)
    hval = packed & ((1 << HALF_NORMAL_BITS) - 1)
    nx, ny, nz = struct.unpack("<eee", hval.to_bytes(6, "little"))
    normal = _safe_normalize((nx, ny, nz))
    rough = _dequant_unit((packed >> HALF_NORMAL_BITS) & ((1 << ROUGH_BITS) - 1),
                          ROUGH_BITS)
    mat = (packed >> (HALF_NORMAL_BITS + ROUGH_BITS)) & ((1 << MAT_BITS) - 1)
    return {"normal": normal, "rough": rough, "mat": mat}


# ---------------------------------------------------------------------------
# 按布局对齐解码
# ---------------------------------------------------------------------------

def _layout_tag(layout):
    """布局指纹：防止打包与布局不一致时被静默解码。"""
    return (int(layout["normal_bits"]) * 3
            + int(layout["rough_bits"]) * 5
            + int(layout["material_bits"]) * 7
            + 0xA5) & 0xFF


def pack_aligned(normal, rough, mat, layout):
    """按指定布局打包：负载高位附加 TAG_BITS 位布局标签。"""
    payload = _pack_payload(normal, rough, mat,
                            int(layout["normal_bits"]),
                            int(layout["rough_bits"]),
                            int(layout["material_bits"]))
    return (_layout_tag(layout) << _layout_total_bits(layout)) | payload


def decode_aligned(packed, layout):
    """按布局解码：返回 dict(normal, rough, mat)；位宽不一致必须报错。

    超出布局总位宽的位（含标签不符）一律报错，绝不静默取零或污染通道。
    """
    total = _layout_total_bits(layout)
    if not isinstance(packed, int) or isinstance(packed, bool) \
            or packed < 0 or packed >= (1 << (total + TAG_BITS)):
        raise ValueError(
            "打包位宽与布局不一致：值超出 %d 位（负载 %d 位 + 标签 %d 位）"
            % (total + TAG_BITS, total, TAG_BITS))
    tag = packed >> total
    if tag != _layout_tag(layout):
        raise ValueError("布局标签不匹配：打包与布局不一致（拒绝静默取零）")
    payload = packed & ((1 << total) - 1)
    return _unpack_payload(payload,
                           int(layout["normal_bits"]),
                           int(layout["rough_bits"]),
                           int(layout["material_bits"]))
