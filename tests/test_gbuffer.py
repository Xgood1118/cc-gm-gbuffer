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
