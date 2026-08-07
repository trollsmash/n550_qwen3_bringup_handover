# 如何获得 w.bin（1.11 GB）

本仓库**不包含**权重文件：GitHub 单文件上限 100 MB，而且它是可完全复现的
产物，不该进版本库。

```
文件名   w.bin
大小     1192100096 字节
md5      2ea6dfa56c710fc389ec32ffd5f62cdb
加载地址 0x88000000
```

**拿到之后先核对 md5**。1.11 GB 经 U 盘或网络传输，损坏不会有任何提示，
上板后表现为"权重 magic 不对"，或者更糟 —— 算出一堆垃圾还不报错。

```
md5sum w.bin                                  # Linux / WSL
certutil -hashfile w.bin MD5                  # Windows
```

## 途径一：找软件侧要现成的（推荐）

直接拷贝，省去下载和转换。核对上面的 md5 即可。

## 途径二：自己从头生成

需要能访问 HuggingFace（或其镜像）。在 Qwen3 项目仓库里：

```bash
source tools/env.sh
python3 tools/02_export_weights.py            # 下载 Qwen3-0.6B 并导出为 BF16 平铺格式
# 产物: golden/qwen3-0.6b-bf16.bin
```

导出是确定性的：同样的模型 + 同样的脚本必然得到逐字节相同的结果，
所以生成完 md5 应当与上面一致。**不一致就说明模型版本不同**，别硬着头皮往下走。

## 格式说明

自定义平铺格式，头部魔数 `QW3M`，其后按 embed_tokens、28 层权重、
final_norm 的顺序依次排列，全部 BF16 小端。程序启动时会校验魔数与
形状常量，用错文件会立刻报错而不是算出垃圾。
