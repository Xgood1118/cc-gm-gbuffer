# -*- coding: utf-8 -*-
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import gbuffer as mod  # noqa: E402
gbuffer = mod


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _random_unit(rnd):
    n = (rnd.gauss(0.0, 1.0), rnd.gauss(0.0, 1.0), rnd.gauss(0.0, 1.0))
    length = math.sqrt(sum(v * v for v in n))
    return tuple(v / length for v in n)


def _layout_is_valid():
    return (
        gbuffer.AZIMUTH_BITS + gbuffer.POLAR_BITS == gbuffer.NORMAL_BITS
        and gbuffer.NORMAL_BITS + gbuffer.MATERIAL_BITS + gbuffer.ROUGH_BITS
        == gbuffer.TOTAL_BITS
    )


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------


def test_layout_fits_exactly_32_bits():
    assert _layout_is_valid()
    assert gbuffer.TOTAL_BITS == 32
    assert gbuffer.NORMAL_BITS == gbuffer.AZIMUTH_BITS + gbuffer.POLAR_BITS


def test_packed_stays_within_total_width():
    rnd = random.Random(7)
    for _ in range(1000):
        n = _random_unit(rnd)
        packed = gbuffer.pack_pixel(n, rnd.random(), rnd.randrange(gbuffer.MAT_MAX + 1))
        assert isinstance(packed, int)
        assert 0 <= packed < (1 << gbuffer.TOTAL_BITS)


# ---------------------------------------------------------------------------
# 原有测试的等价覆盖
# ---------------------------------------------------------------------------


def test_roundtrip_material():
    px = gbuffer.GBuffer((0.0, 0.0, 1.0), 0.5, 7)
    packed = gbuffer.pack_pixel(px.normal, px.rough, px.mat)
    back = gbuffer.unpack_pixel(packed)
    assert back["mat"] == 7


def test_normal_error_bounded_basic():
    px = gbuffer.GBuffer((0.0, 0.0, 1.0), 0.5, 3)
    packed = gbuffer.pack_pixel(px.normal, px.rough, px.mat)
    back = gbuffer.unpack_pixel(packed)
    err = gbuffer.measure_error(px, back)
    assert err["angle_deg"] < 5.0


# ---------------------------------------------------------------------------
# 法线角度误差上界
# ---------------------------------------------------------------------------


def test_normal_angle_error_bounded_everywhere():
    """随机 + 特殊方向上，角度误差都不超过模块声明的理论上界。"""
    rnd = random.Random(2024)
    special = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
    ]
    samples = special + [_random_unit(rnd) for _ in range(5000)]
    for normal in samples:
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel(normal, 0.3, 1))
        err = gbuffer.measure_error(gbuffer.GBuffer(normal, 0.3, 1), back)
        assert err["angle_deg"] <= gbuffer.NORMAL_MAX_ANGLE_DEG + 1e-12, normal
        assert err["angle_rad"] <= gbuffer.NORMAL_MAX_ANGLE_RAD + 1e-12, normal


def test_unpacked_normal_is_renormalized():
    rnd = random.Random(11)
    for _ in range(2000):
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel(_random_unit(rnd), 0.5, 0))
        length = math.sqrt(sum(v * v for v in back["normal"]))
        assert abs(length - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 粗糙度误差上界
# ---------------------------------------------------------------------------


def test_rough_error_bounded():
    rnd = random.Random(33)
    values = [0.0, 1.0, 0.5] + [rnd.random() for _ in range(3000)]
    for rough in values:
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.0, 0.0, 1.0), rough, 0))
        assert 0.0 <= back["rough"] <= 1.0
        err = gbuffer.measure_error(gbuffer.GBuffer((0.0, 0.0, 1.0), rough, 0), back)
        assert err["rough_abs"] <= gbuffer.ROUGH_MAX_ERROR + 1e-12, rough


# ---------------------------------------------------------------------------
# 材质编号无损
# ---------------------------------------------------------------------------


def test_material_lossless_full_range():
    for mat in range(gbuffer.MAT_MAX + 1):
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.0, mat))
        assert back["mat"] == mat


def test_material_lossless_with_extreme_other_channels():
    for normal in [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1e-9, 0.0, 1.0)]:
        for rough in (0.0, 1.0):
            for mat in (0, 1, 127, 255):
                back = gbuffer.unpack_pixel(gbuffer.pack_pixel(normal, rough, mat))
                assert back["mat"] == mat


# ---------------------------------------------------------------------------
# 通道互不污染
# ---------------------------------------------------------------------------


def test_changing_material_does_not_change_other_channels():
    base = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, -0.4, 0.5), 0.37, 0))
    for mat in (1, 42, gbuffer.MAT_MAX):
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, -0.4, 0.5), 0.37, mat))
        assert back["normal"] == base["normal"]
        assert back["rough"] == base["rough"]
        assert back["mat"] == mat


def test_changing_rough_does_not_change_normal_or_material():
    base = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, -0.4, 0.5), 0.0, 9))
    for rough in (0.0, 0.25, 0.5, 1.0):
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.3, -0.4, 0.5), rough, 9))
        assert back["normal"] == base["normal"]
        assert back["mat"] == 9


def test_changing_normal_does_not_change_rough_or_material():
    base = gbuffer.unpack_pixel(gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.61, 21))
    for normal in [(1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.6, 0.8, 0.0)]:
        back = gbuffer.unpack_pixel(gbuffer.pack_pixel(normal, 0.61, 21))
        assert back["rough"] == base["rough"]
        assert back["mat"] == 21


def test_bit_fields_do_not_bleed():
    """翻转某个字段的任意一位，另外两个通道必须逐位不变。"""
    packed0 = gbuffer.pack_pixel((0.6, 0.8, 0.0), 0.5, 3)
    base = gbuffer.unpack_pixel(packed0)

    groups = {
        "rough": [(gbuffer.ROUGH_SHIFT, gbuffer.ROUGH_BITS)],
        "mat": [(gbuffer.MATERIAL_SHIFT, gbuffer.MATERIAL_BITS)],
        "normal": [
            (gbuffer.AZIMUTH_SHIFT, gbuffer.AZIMUTH_BITS),
            (gbuffer.POLAR_SHIFT, gbuffer.POLAR_BITS),
        ],
    }
    for owner, spans in groups.items():
        for shift, width in spans:
            for bit in range(width):
                flipped = gbuffer.unpack_pixel(packed0 ^ (1 << (shift + bit)))
                for key in ("normal", "rough", "mat"):
                    if key != owner:
                        assert flipped[key] == base[key], (owner, bit, key)


# ---------------------------------------------------------------------------
# 边界输入
# ---------------------------------------------------------------------------


def test_zero_length_normal_has_deterministic_result():
    first = gbuffer.pack_pixel((0.0, 0.0, 0.0), 0.5, 11)
    second = gbuffer.pack_pixel((0.0, 0.0, 0.0), 0.5, 11)
    assert first == second

    back = gbuffer.unpack_pixel(first)
    assert back["mat"] == 11
    assert abs(back["normal"][2] - 1.0) < 1e-9
    err = gbuffer.measure_error(gbuffer.GBuffer((0.0, 0.0, 0.0), 0.5, 11), back)
    assert err["angle_deg"] == 0.0


def test_non_unit_normal_is_normalized():
    a = gbuffer.pack_pixel((0.0, 0.0, 10.0), 0.5, 4)
    b = gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.5, 4)
    assert a == b


def test_non_finite_normal_is_rejected():
    for normal in [
        (float("nan"), 0.0, 1.0),
        (0.0, float("inf"), 1.0),
        (0.0, 0.0, float("-inf")),
    ]:
        with pytest.raises(ValueError):
            gbuffer.pack_pixel(normal, 0.5, 0)


def test_rough_out_of_range_is_clamped():
    cases = [
        (-1.0, 0.0),
        (-1e-9, 0.0),
        (1.5, 1.0),
        (1e9, 1.0),
        (float("-inf"), 0.0),
        (float("inf"), 1.0),
    ]
    for rough, expected in cases:
        packed = gbuffer.pack_pixel((0.0, 0.0, 1.0), rough, 5)
        back = gbuffer.unpack_pixel(packed)
        assert back["rough"] == expected
        assert back["mat"] == 5
        assert packed == gbuffer.pack_pixel((0.0, 0.0, 1.0), expected, 5)


def test_rough_nan_is_rejected():
    with pytest.raises(ValueError):
        gbuffer.pack_pixel((0.0, 0.0, 1.0), float("nan"), 0)


def test_material_out_of_range_is_clamped():
    cases = [(-1, 0), (-1000, 0), (gbuffer.MAT_MAX + 1, gbuffer.MAT_MAX), (10**9, gbuffer.MAT_MAX)]
    for mat, expected in cases:
        packed = gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.5, mat)
        back = gbuffer.unpack_pixel(packed)
        assert back["mat"] == expected
        assert back["rough"] == gbuffer.unpack_pixel(
            gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.5, 0)
        )["rough"]

        err = gbuffer.measure_error(gbuffer.GBuffer((0.0, 0.0, 1.0), 0.5, mat), back)
        assert err["mat_exact"] is False


def test_non_integer_material_is_rejected():
    for mat in (3.5, 3.0, "3", None):
        with pytest.raises(TypeError):
            gbuffer.pack_pixel((0.0, 0.0, 1.0), 0.5, mat)


def test_out_of_range_material_does_not_pollute_other_channels():
    base = gbuffer.pack_pixel((0.2, 0.3, 0.4), 0.42, 7)
    clamped = gbuffer.pack_pixel((0.2, 0.3, 0.4), 0.42, 10**9)
    base_back = gbuffer.unpack_pixel(base)
    clamped_back = gbuffer.unpack_pixel(clamped)
    assert clamped_back["normal"] == base_back["normal"]
    assert clamped_back["rough"] == base_back["rough"]


# ---------------------------------------------------------------------------
# 打包确定性
# ---------------------------------------------------------------------------


def test_pack_is_deterministic():
    rnd = random.Random(99)
    for _ in range(2000):
        normal = _random_unit(rnd)
        rough = rnd.random()
        mat = rnd.randrange(gbuffer.MAT_MAX + 1)
        assert gbuffer.pack_pixel(normal, rough, mat) == gbuffer.pack_pixel(
            normal, rough, mat
        )


def test_pack_is_stable_under_equivalent_inputs():
    """等价输入（未归一化、整数分量）得到相同打包结果。"""
    assert gbuffer.pack_pixel((0.6, 0.8, 0.0), 0.9, 12) == gbuffer.pack_pixel(
        (3.0, 4.0, 0.0), 0.9, 12
    )
    assert gbuffer.pack_pixel((0, 0, 1), 1, 255) == gbuffer.pack_pixel(
        (0.0, 0.0, 1.0), 1.0, 255
    )


# ---------------------------------------------------------------------------
# 解包 / 误差输入校验
# ---------------------------------------------------------------------------


def test_unpack_ignores_bits_above_total_width():
    packed = gbuffer.pack_pixel((0.1, 0.2, 0.9), 0.4, 9)
    assert gbuffer.unpack_pixel(packed | (0xFF << gbuffer.TOTAL_BITS)) == \
        gbuffer.unpack_pixel(packed)


def test_unpack_accepts_two_complement_negative():
    packed = gbuffer.pack_pixel((0.1, 0.2, 0.9), 0.4, 9)
    assert gbuffer.unpack_pixel(packed - (1 << gbuffer.TOTAL_BITS)) == \
        gbuffer.unpack_pixel(packed)


def test_unpack_rejects_non_integer():
    for value in (1.5, "1", None):
        with pytest.raises(TypeError):
            gbuffer.unpack_pixel(value)


def test_measure_error_accepts_dict_and_gbuffer():
    px = gbuffer.GBuffer((0.0, 0.0, 1.0), 0.5, 3)
    back = gbuffer.unpack_pixel(gbuffer.pack_pixel(px.normal, px.rough, px.mat))
    a = gbuffer.measure_error(px, back)
    b = gbuffer.measure_error(
        {"normal": px.normal, "rough": px.rough, "mat": px.mat}, back
    )
    assert a == b
    assert a["mat_exact"] is True


def test_measure_error_reports_exact_material():
    # 0 与 1 可以被量化精确表示。
    px = gbuffer.GBuffer((0.0, 0.0, 1.0), 1.0, 255)
    back = gbuffer.unpack_pixel(gbuffer.pack_pixel(px.normal, px.rough, px.mat))
    err = gbuffer.measure_error(px, back)
    assert err["mat_exact"] is True
    assert err["angle_deg"] == 0.0
    assert err["rough_abs"] == 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_stable_output(tmp_path):
    from gbuffer import cli

    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "pixels.json")
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    cli.run(data, str(out1))
    cli.run(data, str(out2))
    assert out1.read_bytes() == out2.read_bytes()

    import json

    result = json.loads(out1.read_text(encoding="utf-8"))
    assert result["stats"]["sample_count"] == 2
    assert result["stats"]["material_roundtrip_ok"] == 2
    assert result["stats"]["out_of_range_samples"] == 0
    assert result["stats"]["max_normal_angle_deg"] <= gbuffer.NORMAL_MAX_ANGLE_DEG
    assert result["stats"]["max_rough_abs_error"] <= gbuffer.ROUGH_MAX_ERROR
