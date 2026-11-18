# -*- coding: utf-8 -*-
"""CLI 验收：打包 data/pixels.json，输出误差实测与布局协商结果。"""

import argparse
import json
import math

import gbuffer as m


def _fibonacci_sphere(count):
    """确定性的球面均匀采样方向，用于实测法线角度误差。"""
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        z = 1.0 - 2.0 * (i + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        pts.append((r * math.cos(theta), r * math.sin(theta), z))
    return pts


def _angle_deg(a, b):
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def _survey(pack, decode, count):
    """对 count 个采样方向做打包-解包，返回实测最大角度误差（度）。"""
    worst = 0.0
    for n in _fibonacci_sphere(count):
        back = decode(pack(n, 0.5, 0))["normal"]
        worst = max(worst, _angle_deg(n, back))
    return worst


def main():
    ap = argparse.ArgumentParser(description="G-Buffer 打包与精度")
    ap.add_argument("input", help="输入像素 JSON（法线/粗糙度/材质编号）")
    ap.add_argument("--out", required=True, help="输出打包结果 JSON")
    ap.add_argument("--samples", type=int, default=20000, help="误差实测采样方向数")
    a = ap.parse_args()

    with open(a.input, "r", encoding="utf-8") as f:
        pixels = json.load(f)

    out_pixels = []
    max_ang32 = max_angh = max_rough = 0.0
    for p in pixels:
        g = m.GBuffer(tuple(p["normal"]), float(p["rough"]), int(p["mat"]))
        packed = m.pack_pixel(g.normal, g.rough, g.mat)
        packed_half = m.pack_pixel_half(g.normal, g.rough, g.mat)
        # 输出确定性：同一输入两次打包必须一致
        assert m.pack_pixel(g.normal, g.rough, g.mat) == packed
        assert m.pack_pixel_half(g.normal, g.rough, g.mat) == packed_half
        e32 = m.measure_error(g, m.unpack_pixel(packed))
        eh = m.measure_error(g, m.decode_half(packed_half))
        max_ang32 = max(max_ang32, e32["angle_deg"])
        max_angh = max(max_angh, eh["angle_deg"])
        max_rough = max(max_rough, e32["rough_err"], eh["rough_err"])
        out_pixels.append({"packed": packed, "packed_half": packed_half})

    survey32 = _survey(m.pack_pixel, m.unpack_pixel, a.samples)
    surveyh = _survey(m.pack_pixel_half, m.decode_half, a.samples)

    candidates = [
        {"normal_bits": 16, "rough_bits": 8, "material_bits": 8},
        {"normal_bits": 20, "rough_bits": 6, "material_bits": 6},
        {"normal_bits": 10, "rough_bits": 11, "material_bits": 11},
        {"normal_bits": 40, "rough_bits": 16, "material_bits": 16},
    ]
    neg = m.negotiate_layout(candidates, max_error_deg=2.0, max_bytes=8)

    result = {
        "layout_32": {"normal_bits": m.NORMAL_BITS, "rough_bits": m.ROUGH_BITS,
                      "material_bits": m.MAT_BITS},
        "pixels": out_pixels,
        "stats": {
            "bound_32_deg": m.normal_error_bound_deg(m.NORMAL_BITS),
            "bound_half_deg": m.HALF_NORMAL_ERROR_BOUND_DEG,
            "rough_bound": m.rough_error_bound(m.ROUGH_BITS),
            "file_max_angle_32_deg": max_ang32,
            "file_max_angle_half_deg": max_angh,
            "file_max_rough_err": max_rough,
            "survey_max_angle_32_deg": survey32,
            "survey_max_angle_half_deg": surveyh,
            "survey_samples": a.samples,
        },
        "negotiation": {
            "candidates": candidates,
            "max_error_deg": 2.0,
            "max_bytes": 8,
            "selected": neg["selected"],
            "rejected": [[i, r] for i, r in neg["rejected"]],
        },
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    s = result["stats"]
    print("== G-Buffer 验收 ==")
    print("像素数: %d -> %s" % (len(out_pixels), a.out))
    print("32 位整包:  法线误差上界 %.4f° | 文件实测 %.4f° | 采样实测 %.4f°"
          % (s["bound_32_deg"], max_ang32, survey32))
    print("半精度整包: 法线误差上界 %.4f° | 文件实测 %.4f° | 采样实测 %.4f°"
          % (s["bound_half_deg"], max_angh, surveyh))
    print("粗糙度: 上界 %.6f | 文件实测 %.6f" % (s["rough_bound"], max_rough))
    print("布局协商: selected=%s rejected=%s" % (neg["selected"], neg["rejected"]))


if __name__ == "__main__":
    main()
