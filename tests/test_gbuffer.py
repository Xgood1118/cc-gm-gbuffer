# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gbuffer as mod  # noqa: E402


def test_roundtrip_material():
    px = gbuffer.GBuffer((0.0, 0.0, 1.0), 0.5, 7)
    packed = gbuffer.pack_pixel(px.normal, px.rough, px.mat)
    back = gbuffer.unpack_pixel(packed)
    assert back["mat"] == 7


def test_normal_error_bounded():
    px = gbuffer.GBuffer((0.0, 0.0, 1.0), 0.5, 3)
    packed = gbuffer.pack_pixel(px.normal, px.rough, px.mat)
    back = gbuffer.unpack_pixel(packed)
    err = gbuffer.measure_error(px, back)
    assert err["angle_deg"] < 5.0


# ===== v2 契约测试：初始全部失败，实现后必须通过 =====
import gbuffer  # noqa: E402



def test_negotiate_picks_feasible():
    cands = [
        {"normal_bits": 16, "rough_bits": 8, "material_bits": 8},
        {"normal_bits": 20, "rough_bits": 6, "material_bits": 6},
        {"normal_bits": 10, "rough_bits": 11, "material_bits": 11},
    ]
    r = gbuffer.negotiate_layout(cands, max_error_deg=2.0, max_bytes=8)
    assert r["selected"] is not None
    assert isinstance(r["rejected"], list)


def test_negotiate_rejects_over_bandwidth():
    r = gbuffer.negotiate_layout([{"normal_bits": 40, "rough_bits": 16, "material_bits": 16}],
                                 max_error_deg=1.0, max_bytes=8)
    assert r["selected"] is None
    assert r["rejected"]


def test_half_normal_error_bound():
    import math
    dirs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1 / math.sqrt(3),) * 3, (0.6, 0.8, 0.0)]
    worst = 0.0
    for n in dirs:
        back = gbuffer.decode_half(gbuffer.pack_pixel_half(n, 0.5, 2))["normal"]
        dot = sum(a * b for a, b in zip(n, back))
        worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
    assert worst <= 3.0


def test_decode_half_material_and_rough():
    d = gbuffer.decode_half(gbuffer.pack_pixel_half((0.0, 0.0, 1.0), 0.75, 7))
    assert d["mat"] == 7
    assert abs(d["rough"] - 0.75) <= 3


def test_decode_aligned_rejects_mismatch():
    try:
        gbuffer.decode_aligned(0xFFFF, {"normal_bits": 10, "rough_bits": 5, "material_bits": 5})
    except Exception as exc:
        assert str(exc)
    else:
        raise AssertionError("位宽不一致时必须报错")


def test_bandwidth_cost_scales():
    lay = {"normal_bits": 16, "rough_bits": 8, "material_bits": 8}
    assert gbuffer.bandwidth_cost(lay, 1920, 1080) == 1920 * 1080 * 4
    assert gbuffer.bandwidth_cost(lay, 1, 1) == 4


# ===== 本轮新增契约测试 =====
import math  # noqa: E402

import pytest  # noqa: E402


# ---- 打包：法线固定位数、解包重新归一化 ----

def test_unpacked_normal_is_unit():
    back = gbuffer.unpack_pixel(gbuffer.pack_pixel((1.0, 2.0, 3.0), 0.5, 1))
    n = back["normal"]
    assert abs(math.sqrt(sum(c * c for c in n)) - 1.0) < 1e-9


def test_normal_error_within_bound_32():
    bound = gbuffer.normal_error_bound_deg(gbuffer.NORMAL_BITS)
    worst = 0.0
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(2000):
        z = 1.0 - 2.0 * (i + 0.5) / 2000
        r = math.sqrt(max(0.0, 1.0 - z * z))
        n = (r * math.cos(golden * i), r * math.sin(golden * i), z)
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel(n, 0.5, 0))["normal"]
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(n, back))))
        worst = max(worst, math.degrees(math.acos(dot)))
    assert worst <= bound


# ---- 精度：通道互不污染、材质精确往返 ----

def test_channels_do_not_pollute():
    ref = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, 0.4, 0.5), 0.33, 7))
    for m in (0, 1, 128, 255):
        for r in (0.0, 0.5, 1.0):
            back = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, 0.4, 0.5), r, m))
            assert back["normal"] == ref["normal"]  # 法线不受粗糙度/材质影响
    # 材质变化不影响粗糙度
    a = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, 0.4, 0.5), 0.6, 0))
    b = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, 0.4, 0.5), 0.6, 255))
    assert a["rough"] == b["rough"]


def test_material_exact_roundtrip_full_range():
    for m in range(256):
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.5, m))
        assert back["mat"] == m


def test_rough_error_bound():
    bound = gbuffer.rough_error_bound(gbuffer.ROUGH_BITS)
    worst = 0.0
    for i in range(501):
        r = i / 500.0
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.0, 0.0, 1.0), r, 0))
        worst = max(worst, abs(back["rough"] - r))
    assert worst <= bound + 1e-12


# ---- 边界：零法线 / 粗糙度越界 / 材质越界 ----

def test_zero_length_normal_deterministic():
    a = gbuffer.pack_pixel((0.0, 0.0, 0.0), 0.5, 1)
    b = gbuffer.pack_pixel((0.0, 0.0, 0.0), 0.5, 1)
    assert a == b
    assert gbuffer.unpack_pixel(a)["normal"] == (0.0, 0.0, 1.0)


def test_rough_out_of_range_clamped():
    lo = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.0, 0.0, 1.0), -0.5, 1))
    hi = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.0, 0.0, 1.0), 1.5, 1))
    assert lo["rough"] == 0.0
    assert hi["rough"] == 1.0


def test_mat_out_of_range_raises():
    for bad in (-1, 256, 10 ** 6):
        with pytest.raises(ValueError):
            gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.5, bad)
        with pytest.raises(ValueError):
            gbuffer.pack_pixel_half((0.0, 0.0, 1.0), 0.5, bad)


# ---- 输出：两次打包一致 ----

def test_pack_deterministic():
    args = ((0.6, 0.8, 0.0), 0.37, 42)
    assert gbuffer.pack_pixel(*args) == gbuffer.pack_pixel(*args)
    assert gbuffer.pack_pixel_half(*args) == gbuffer.pack_pixel_half(*args)


# ---- 位宽不足时的优先级分配 ----

def test_allocate_bits_priority_material_first():
    alloc = gbuffer.allocate_bits(10, rough_want=8, mat_want=8)
    assert alloc["material_bits"] == 8  # 材质优先拿满
    assert alloc["rough_bits"] == 2     # 粗糙度分剩余
    alloc = gbuffer.allocate_bits(4, rough_want=8, mat_want=8)
    assert alloc["material_bits"] == 4
    assert alloc["rough_bits"] == 0


# ---- 布局协商：理由码 ----

def test_negotiate_reason_codes():
    cands = [
        {"normal_bits": 4, "rough_bits": 4, "material_bits": 4},    # 误差超限
        {"normal_bits": 64, "rough_bits": 16, "material_bits": 16},  # 带宽超限
        {"normal_bits": 16, "rough_bits": 8, "material_bits": 8},    # 可行
    ]
    r = gbuffer.negotiate_layout(cands, max_error_deg=2.0, max_bytes=8)
    assert r["selected"] == 2
    reasons = dict(r["rejected"])
    assert reasons[0] == gbuffer.REASON_ERROR
    assert reasons[1] == gbuffer.REASON_BANDWIDTH


def test_negotiate_none_feasible():
    r = gbuffer.negotiate_layout(
        [{"normal_bits": 2, "rough_bits": 1, "material_bits": 1}],
        max_error_deg=0.5, max_bytes=8)
    assert r["selected"] is None
    assert r["rejected"] == [(0, gbuffer.REASON_ERROR)]


# ---- 半精度：上界更紧、实测值 ----

def test_half_bound_tighter_than_32bit():
    assert gbuffer.HALF_NORMAL_ERROR_BOUND_DEG < gbuffer.normal_error_bound_deg(
        gbuffer.NORMAL_BITS)


def test_half_measured_within_bound():
    worst = 0.0
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(2000):
        z = 1.0 - 2.0 * (i + 0.5) / 2000
        r = math.sqrt(max(0.0, 1.0 - z * z))
        n = (r * math.cos(golden * i), r * math.sin(golden * i), z)
        back = gbuffer.decode_half(gbuffer.pack_pixel_half(n, 0.5, 0))["normal"]
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(n, back))))
        worst = max(worst, math.degrees(math.acos(dot)))
    assert worst <= gbuffer.HALF_NORMAL_ERROR_BOUND_DEG


def test_half_zero_normal_and_clamp():
    d = gbuffer.decode_half(gbuffer.pack_pixel_half((0.0, 0.0, 0.0), 2.0, 3))
    assert d["normal"] == (0.0, 0.0, 1.0)
    assert d["rough"] == 1.0
    assert d["mat"] == 3


# ---- 解码对齐 ----

def test_decode_aligned_roundtrip():
    layout = {"normal_bits": 16, "rough_bits": 8, "material_bits": 8}
    packed = gbuffer.pack_aligned((0.6, 0.8, 0.0), 0.5, 7, layout)
    back = gbuffer.decode_aligned(packed, layout)
    assert back["mat"] == 7
    assert abs(back["rough"] - 0.5) <= gbuffer.rough_error_bound(8) + 1e-12
    dot = sum(a * b for a, b in zip((0.6, 0.8, 0.0), back["normal"]))
    assert math.degrees(math.acos(max(-1.0, min(1.0, dot)))) < 1.0


def test_decode_aligned_extra_bits_rejected():
    layout = {"normal_bits": 16, "rough_bits": 8, "material_bits": 8}
    packed = gbuffer.pack_aligned((0.0, 0.0, 1.0), 0.5, 7, layout)
    total = 32 + gbuffer.TAG_BITS
    with pytest.raises(ValueError):
        gbuffer.decode_aligned(packed | (1 << (total + 2)), layout)


def test_decode_aligned_wrong_layout_rejected():
    layout = {"normal_bits": 16, "rough_bits": 8, "material_bits": 8}
    packed = gbuffer.pack_aligned((0.0, 0.0, 1.0), 0.5, 7, layout)
    other = {"normal_bits": 12, "rough_bits": 10, "material_bits": 10}
    with pytest.raises(ValueError):
        gbuffer.decode_aligned(packed, other)


def test_decode_aligned_untagged_payload_rejected():
    # 裸 32 位整包（无布局标签）不得被静默解码
    layout = {"normal_bits": 16, "rough_bits": 8, "material_bits": 8}
    payload = gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.5, 7)
    with pytest.raises(ValueError):
        gbuffer.decode_aligned(payload, layout)


def test_decode_aligned_no_silent_zero():
    # 标签错误的包必须报错，而不是把通道静默取零
    layout = {"normal_bits": 16, "rough_bits": 8, "material_bits": 8}
    packed = gbuffer.pack_aligned((0.0, 0.0, 1.0), 0.9, 200, layout)
    forged = packed ^ (0xFF << 32)  # 翻转标签位
    with pytest.raises(ValueError):
        gbuffer.decode_aligned(forged, layout)
