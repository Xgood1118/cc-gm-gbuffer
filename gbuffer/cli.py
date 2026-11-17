# -*- coding: utf-8 -*-
"""G-Buffer 命令行工具：打包 JSON 中的像素并统计误差。"""

import argparse
import json

import gbuffer as m


def _effective_input(pixel):
    """返回用于判断输入是否越界的规整结果。"""
    zero_normal = False
    try:
        nx, ny, nz = pixel["normal"]
        zero_normal = all(float(v) == 0.0 for v in (nx, ny, nz))
    except (KeyError, TypeError, ValueError):
        pass

    rough = pixel.get("rough")
    rough_out = False
    try:
        rough_out = not (0.0 <= float(rough) <= 1.0)
    except (TypeError, ValueError):
        rough_out = True

    mat = pixel.get("mat")
    mat_out = False
    try:
        value = mat.__index__()
        mat_out = not (m.MATERIAL_RANGE[0] <= value <= m.MATERIAL_RANGE[1])
    except AttributeError:
        mat_out = True

    return zero_normal, rough_out, mat_out


def run(inp, out):
    with open(inp, encoding="utf-8") as f:
        pixels = json.load(f)
    if not isinstance(pixels, list):
        raise SystemExit("输入必须是像素对象数组")

    rows = []
    max_angle = 0.0
    max_rough = 0.0
    mat_roundtrip = 0
    out_of_range = 0

    for index, pixel in enumerate(pixels):
        normal = pixel["normal"]
        rough = pixel["rough"]
        mat = pixel["mat"]

        prev = m.GBuffer(normal=tuple(normal), rough=rough, mat=mat)
        packed = m.pack_pixel(normal, rough, mat)
        # 打包幂等性：同样的输入两次得到同样的整数。
        packed_again = m.pack_pixel(normal, rough, mat)
        if packed != packed_again:
            raise SystemExit("内部错误：第 %d 个像素打包结果不稳定" % index)

        back = m.unpack_pixel(packed)
        err = m.measure_error(prev, back)

        zero_normal, rough_out, mat_out = _effective_input(pixel)
        if zero_normal or rough_out or mat_out:
            out_of_range += 1
        if err["mat_exact"]:
            mat_roundtrip += 1
        max_angle = max(max_angle, err["angle_deg"])
        max_rough = max(max_rough, err["rough_abs"])

        rows.append(
            {
                "index": index,
                "packed": packed,
                "packed_hex": "0x%08x" % packed,
                "unpacked": {
                    "normal": [round(v, 12) for v in back["normal"]],
                    "rough": round(back["rough"], 12),
                    "mat": back["mat"],
                },
                "error": {
                    "angle_deg": err["angle_deg"],
                    "rough_abs": err["rough_abs"],
                    "mat_exact": err["mat_exact"],
                },
            }
        )

    result = {
        "format": {
            "total_bits": m.TOTAL_BITS,
            "normal_bits": m.NORMAL_BITS,
            "azimuth_bits": m.AZIMUTH_BITS,
            "polar_bits": m.POLAR_BITS,
            "material_bits": m.MATERIAL_BITS,
            "rough_bits": m.ROUGH_BITS,
            "normal_max_angle_deg": m.NORMAL_MAX_ANGLE_DEG,
            "rough_max_error": m.ROUGH_MAX_ERROR,
        },
        "stats": {
            "sample_count": len(pixels),
            "max_normal_angle_deg": max_angle,
            "max_rough_abs_error": max_rough,
            "material_roundtrip_ok": mat_roundtrip,
            "material_roundtrip_total": len(pixels),
            "out_of_range_samples": out_of_range,
        },
        "pixels": rows,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    stats = result["stats"]
    print("样本数:                    %d" % stats["sample_count"])
    print("最大法线角度误差:          %.6f deg" % stats["max_normal_angle_deg"])
    print("法线角度误差上界:          %.6f deg" % m.NORMAL_MAX_ANGLE_DEG)
    print("最大粗糙度误差:            %.6f" % stats["max_rough_abs_error"])
    print("粗糙度误差上界:            %.6f" % m.ROUGH_MAX_ERROR)
    print(
        "材质编号往返一致:          %d / %d"
        % (stats["material_roundtrip_ok"], stats["material_roundtrip_total"])
    )
    print("越界样本数:                %d" % stats["out_of_range_samples"])
    print("已写入:                    %s" % out)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="G-Buffer 打包与精度")
    ap.add_argument("input", help="输入像素 JSON 文件")
    ap.add_argument("--out", required=True, help="输出打包结果 JSON 文件")
    a = ap.parse_args(argv)
    run(a.input, a.out)


if __name__ == "__main__":
    main()
