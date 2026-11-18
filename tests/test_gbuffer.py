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
