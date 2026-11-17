# G-Buffer 打包与精度

一个把法线、粗糙度、材质编号打包进有限位宽、并能无损解包的 G-Buffer 编解码器

## 保留签名

- `GBuffer`
- `pack_pixel(normal, rough, mat)`
- `unpack_pixel(packed)`
- `measure_error(orig, back)`

## 自测

```
python -m pytest tests -q
python -m gbuffer.cli data/pixels.json --out packed.json
```
