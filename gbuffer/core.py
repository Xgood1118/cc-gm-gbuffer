# -*- coding: utf-8 -*-
"""G-Buffer 打包与精度。

本模块把法线、粗糙度、材质编号打包进一个固定的 32 位整数。位布局如下
（从低位到高位）::

    位  0..5    粗糙度      ROUGH_BITS   = 6
    位  6..13   材质编号    MATERIAL_BITS = 8
    位 14..22   法线方位角  AZIMUTH_BITS  = 9
    位 23..31   法线极角    POLAR_BITS    = 9

法线占用固定的 18 位；剩余 14 位由材质编号与粗糙度共用。分配优先级是
**材质编号优先**：材质编号必须无损往返，所以先拿满 8 位，粗糙度得到余下
的 6 位。若总位宽被调小到连材质编号都放不下，模块导入时就会报错，而不是
悄悄量化材质编号。

法线使用球面角量化：解包时按方位角/极角重建单位向量，并显式重新归一化。
"""

from dataclasses import dataclass
import math

# ---------------------------------------------------------------------------
# 位宽与优先级
# ---------------------------------------------------------------------------

TOTAL_BITS = 32

# 法线固定占用位数。
AZIMUTH_BITS = 9
POLAR_BITS = 9
NORMAL_BITS = AZIMUTH_BITS + POLAR_BITS  # 18

# 材质编号的无损编码所需位数。
REQUESTED_MATERIAL_BITS = 8


def _allocate_remaining_bits(total_bits, normal_bits, material_bits):
    """在法线之后分配剩余位。

    优先级：材质编号（无损）优先于粗糙度。返回 ``(material_bits,
    rough_bits)``。如果剩余位连材质编号都放不下，抛出 ``ValueError``，
    因为此时无法保证材质编号无损。
    """
    remaining = total_bits - normal_bits
    if remaining < material_bits:
        raise ValueError(
            "位宽不足：法线占 %d 位后只剩 %d 位，材质编号需要 %d 位才能无损编码"
            % (normal_bits, remaining, material_bits)
        )
    return material_bits, remaining - material_bits


MATERIAL_BITS, ROUGH_BITS = _allocate_remaining_bits(
    TOTAL_BITS, NORMAL_BITS, REQUESTED_MATERIAL_BITS
)

ROUGH_SHIFT = 0
MATERIAL_SHIFT = ROUGH_SHIFT + ROUGH_BITS
AZIMUTH_SHIFT = MATERIAL_SHIFT + MATERIAL_BITS
POLAR_SHIFT = AZIMUTH_SHIFT + AZIMUTH_BITS

ROUGH_MASK = (1 << ROUGH_BITS) - 1
MATERIAL_MASK = (1 << MATERIAL_BITS) - 1
AZIMUTH_MASK = (1 << AZIMUTH_BITS) - 1
POLAR_MASK = (1 << POLAR_BITS) - 1
PACKED_MASK = (1 << TOTAL_BITS) - 1

MAT_MAX = MATERIAL_MASK

# 材质编号的有效范围；越界时饱和到边界（见 pack_pixel）。
MATERIAL_RANGE = (0, MAT_MAX)

_FALLBACK_NORMAL = (0.0, 0.0, 1.0)
_TWO_PI = 2.0 * math.pi


def _polar_levels():
    if POLAR_BITS <= 0:
        return 0
    return (1 << POLAR_BITS) - 1


def _rough_levels():
    if ROUGH_BITS <= 0:
        return 0
    return (1 << ROUGH_BITS) - 1


# 理论上界（不是经验值）：
#   方位角误差 <= 半步 = pi / 2^AZIMUTH_BITS
#   极角误差   <= 半步 = pi / (2 * (2^POLAR_BITS - 1))
# 球面上两点夹角 <= |d phi| + |d theta|，因此：
NORMAL_MAX_ANGLE_RAD = math.pi / (1 << AZIMUTH_BITS)
NORMAL_MAX_ANGLE_RAD += (
    math.pi if POLAR_BITS <= 0 else math.pi / (2 * _polar_levels())
)
NORMAL_MAX_ANGLE_DEG = math.degrees(NORMAL_MAX_ANGLE_RAD)

ROUGH_MAX_ERROR = 1.0 if ROUGH_BITS <= 0 else 0.5 / _rough_levels()

__all__ = [
    "GBuffer",
    "pack_pixel",
    "unpack_pixel",
    "measure_error",
    "TOTAL_BITS",
    "NORMAL_BITS",
    "AZIMUTH_BITS",
    "POLAR_BITS",
    "MATERIAL_BITS",
    "ROUGH_BITS",
    "ROUGH_SHIFT",
    "MATERIAL_SHIFT",
    "AZIMUTH_SHIFT",
    "POLAR_SHIFT",
    "ROUGH_MASK",
    "MATERIAL_MASK",
    "AZIMUTH_MASK",
    "POLAR_MASK",
    "PACKED_MASK",
    "MAT_MAX",
    "MATERIAL_RANGE",
    "NORMAL_MAX_ANGLE_RAD",
    "NORMAL_MAX_ANGLE_DEG",
    "ROUGH_MAX_ERROR",
]


@dataclass
class GBuffer:
    """一个像素的 G-Buffer 数据。"""

    normal: tuple
    rough: float
    mat: int


# ---------------------------------------------------------------------------
# 输入规整与边界策略
# ---------------------------------------------------------------------------


def _as_finite_float(value, what):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s 必须是数值" % what)
    if not math.isfinite(result):
        raise ValueError("%s 必须是有限数" % what)
    return result


def _normalized_or_fallback(normal):
    """返回 (单位法线, 是否使用了零长度回退)。

    长度为 0 的法线没有方向，确定性地回退到 ``(0, 0, 1)``。非有限输入
    无法给出确定方向，抛出 ``ValueError``。非单位长度输入会被归一化。
    """
    try:
        nx, ny, nz = normal
    except (TypeError, ValueError):
        raise ValueError("法线必须是包含 3 个分量的序列")
    nx = _as_finite_float(nx, "法线分量")
    ny = _as_finite_float(ny, "法线分量")
    nz = _as_finite_float(nz, "法线分量")

    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        return _FALLBACK_NORMAL, True
    return (nx / length, ny / length, nz / length), False


def _effective_rough(rough):
    """粗糙度边界策略：饱和到 [0, 1]。

    NaN 无法排序，抛出 ``ValueError``；+inf/-inf 方向明确，分别饱和到 1/0。
    """
    try:
        value = float(rough)
    except (TypeError, ValueError):
        raise ValueError("粗糙度 必须是数值")
    if math.isnan(value):
        raise ValueError("粗糙度 不能是 NaN")
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _effective_mat(mat):
    """材质编号边界策略：必须是整数，越界饱和到 [0, MAT_MAX]。

    材质编号不做量化：在有效范围内逐位无损往返。越界值确定性地饱和到
    最近的边界，且同一越界输入总是得到同一个结果。
    """
    try:
        value = mat.__index__()
    except AttributeError:
        raise TypeError("材质编号必须是整数")
    if value < 0:
        return 0
    if value > MAT_MAX:
        return MAT_MAX
    return value


# ---------------------------------------------------------------------------
# 量化 / 反量化
# ---------------------------------------------------------------------------


def _quantize_azimuth(unit_normal):
    x, y, _ = unit_normal
    theta = math.atan2(y, x)
    if theta < 0.0:
        theta += _TWO_PI
    scale = 1 << AZIMUTH_BITS
    q = int(math.floor(theta / _TWO_PI * scale + 0.5))
    return q % scale


def _quantize_polar(unit_normal):
    levels = _polar_levels()
    if levels <= 0:
        return 0
    z = unit_normal[2]
    if z < -1.0:
        z = -1.0
    elif z > 1.0:
        z = 1.0
    phi = math.acos(z)
    q = int(math.floor(phi / math.pi * levels + 0.5))
    if q < 0:
        q = 0
    elif q > levels:
        q = levels
    return q


def _quantize_rough(rough):
    levels = _rough_levels()
    if levels <= 0:
        return 0
    q = int(math.floor(rough * levels + 0.5))
    if q < 0:
        q = 0
    elif q > levels:
        q = levels
    return q


def _decode_normal(azimuth_q, polar_q):
    theta = azimuth_q * _TWO_PI / (1 << AZIMUTH_BITS)
    if POLAR_BITS <= 0:
        phi = 0.0
    else:
        phi = polar_q * math.pi / _polar_levels()

    sin_phi = math.sin(phi)
    x = sin_phi * math.cos(theta)
    y = sin_phi * math.sin(theta)
    z = math.cos(phi)

    # 解包后重新归一化，隔离浮点误差，保证返回的始终是单位向量。
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        return _FALLBACK_NORMAL
    return (x / length, y / length, z / length)


def _decode_rough(q):
    levels = _rough_levels()
    if levels <= 0:
        return 0.0
    return q / float(levels)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def pack_pixel(normal, rough, mat):
    """把一个像素打包成 32 位整数。

    边界行为（均确定）：
      * 法线长度为 0 -> 使用 ``(0, 0, 1)``；非有限分量 -> ``ValueError``。
      * 粗糙度越界 -> 饱和到 ``[0, 1]``；NaN -> ``ValueError``。
      * 材质编号越界 -> 饱和到 ``[0, 255]``；非整数 -> ``TypeError``。
    """
    unit_normal, _ = _normalized_or_fallback(normal)
    rough_value = _effective_rough(rough)
    mat_value = _effective_mat(mat)

    packed = (
        (_quantize_rough(rough_value) & ROUGH_MASK) << ROUGH_SHIFT
        | (mat_value & MATERIAL_MASK) << MATERIAL_SHIFT
        | (_quantize_azimuth(unit_normal) & AZIMUTH_MASK) << AZIMUTH_SHIFT
        | (_quantize_polar(unit_normal) & POLAR_MASK) << POLAR_SHIFT
    )

    # 所有字段都已按掩码写入，这里只是把不变量写清楚。
    assert 0 <= packed <= PACKED_MASK
    return packed


def unpack_pixel(packed):
    """解包 32 位整数，返回 ``{"normal": tuple, "rough": float, "mat": int}``。

    超出 32 位的位被忽略（按低 32 位解释），负整数按低 32 位的补码解释。
    每个字段单独掩码，因此字段之间不会相互污染。
    """
    if not isinstance(packed, int):
        raise TypeError("packed 必须是整数")
    value = packed & PACKED_MASK

    rough_q = (value >> ROUGH_SHIFT) & ROUGH_MASK
    mat_q = (value >> MATERIAL_SHIFT) & MATERIAL_MASK
    azimuth_q = (value >> AZIMUTH_SHIFT) & AZIMUTH_MASK
    polar_q = (value >> POLAR_SHIFT) & POLAR_MASK

    return {
        "normal": _decode_normal(azimuth_q, polar_q),
        "rough": _decode_rough(rough_q),
        "mat": mat_q,
    }


def _parts(pixel):
    if isinstance(pixel, GBuffer):
        return pixel.normal, pixel.rough, pixel.mat
    if isinstance(pixel, dict):
        return pixel["normal"], pixel["rough"], pixel["mat"]
    if all(hasattr(pixel, name) for name in ("normal", "rough", "mat")):
        return pixel.normal, pixel.rough, pixel.mat
    raise TypeError("像素必须是 GBuffer、字典或带 normal/rough/mat 属性的对象")


def measure_error(orig, back):
    """测量原始像素与解包像素之间的误差。

    ``orig`` 可以是 :class:`GBuffer` 或字典，``back`` 是
    :meth:`unpack_pixel` 的返回值（或同构对象）。返回字典包含：

      * ``angle_deg`` / ``angle_rad``：法线夹角；
      * ``rough_abs`` / ``rough_error``：粗糙度绝对误差（对原始值按打包时
        的饱和策略取有效值后比较）；
      * ``mat_exact``：材质编号是否完全一致。

    原始法线为零长度时，按打包时的回退法线计算角度。
    """
    orig_normal, orig_rough, orig_mat = _parts(orig)
    back_normal, back_rough, back_mat = _parts(back)

    a, _ = _normalized_or_fallback(orig_normal)
    b, _ = _normalized_or_fallback(back_normal)

    cross_x = a[1] * b[2] - a[2] * b[1]
    cross_y = a[2] * b[0] - a[0] * b[2]
    cross_z = a[0] * b[1] - a[1] * b[0]
    cross_len = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    angle_rad = math.atan2(cross_len, dot)
    angle_deg = math.degrees(angle_rad)

    rough_error = abs(_effective_rough(orig_rough) - _effective_rough(back_rough))

    try:
        back_mat_value = back_mat.__index__()
    except AttributeError:
        back_mat_value = back_mat
    # 与原始（未饱和）值比较：越界输入被饱和后即使等于边界值，
    # 也不应算作“完全往返”。
    mat_exact = orig_mat == back_mat_value

    return {
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "rough_abs": rough_error,
        "rough_error": rough_error,
        "mat_exact": mat_exact,
    }
