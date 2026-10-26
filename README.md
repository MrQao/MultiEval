# MultiEval

一个用于学习多模态模型评测的轻量 Python 项目。以 POPE 物体存在性问答为例，串联数据抽样、Qwen-VL 本地推理、Yes/No 解析、指标计算、错误案例分析和 Gaussian Blur 鲁棒性实验。

> 此仓库保存原始教学项目及历史实验代码。完整审计和修复在独立副本中完成，没有合并到本仓库。历史结果包含解析器和指标口径问题，使用前请阅读下方说明。

## 功能

- 从 Hugging Face 加载 `oplex/POPE`，按类别和标签分层抽样。
- 通过 MLX-VLM 调用 `mlx-community/Qwen3-VL-2B-Instruct-4bit`。
- 使用统一样本接口连接原始数据和图像扰动。
- 保存原始回答、预测、标签、推理时间与运行状态，支持跳过已成功样本。
- 计算总体和分类指标，导出错误案例 HTML 与人工标注。
- 对比 Gaussian Blur 半径 0、1、3、5 下的预测变化。

## 目录

```text
MultiEval/
├── eval_datasets/           # 数据接口、POPE 数据集和扰动包装器
├── models/                  # 模型接口和 Qwen-VL 实现
├── perturbations/           # Gaussian Blur
├── evaluators/              # 解析、存储、指标、错误分析和配对统计
├── visualizations/          # 结果绘图
├── run_pope.py              # POPE 教学实验与演示函数
├── run_robustness.py        # 四种模糊半径实验
├── analyze_robustness.py    # 已有90题实验的离线分析
├── test_pope.py             # 数据查看演示
├── test_pope_qwen.py        # 单样本模型演示
├── test_qwen.py             # 本地图片模型演示
├── data/                    # 本机示例图片（不上传）
└── results/                 # 本机实验输出（不上传）
```

`test_*.py` 是教学演示脚本，执行时可能加载数据或模型，并非自动化单元测试。

## 环境与安装

推荐 Python 3.11，以及能够运行 Metal 的 Apple Silicon Mac。模型推理依赖 MLX；Linux、Windows 或无法访问 Metal 的环境不能直接使用当前模型实现。数据与模型首次使用会从 Hugging Face 下载，请预留网络、磁盘和内存空间。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 列出直接依赖，未锁定版本。它不是历史实验环境的精确复现文件；不同版本可能影响模型加载、生成默认参数与结果。

## 使用方法

所有命令均从项目根目录执行。

### 1. 查看数据

```bash
python test_pope.py
```

### 2. 单题模型推理

```bash
python test_pope_qwen.py
```

`test_qwen.py` 使用本机 `data/test.jpg`；运行前需自行提供该图片。

### 3. POPE 教学实验

`run_pope.py` 提供以下函数：

| 函数 | 用途 |
| --- | --- |
| `main()` | 分层抽样、推理并计算指标 |
| `use_storage()` | 重读结果、错误分析和HTML导出 |
| `blur_test()` | 保存四个半径的模糊示例 |
| `dataset_wrapper()` | 查看原图与扰动图 |

当前直接运行 `python run_pope.py` 默认调用 `dataset_wrapper()`，会打开图片查看器。若要运行推理实验，可显式调用：

```bash
python -c 'from run_pope import main; main()'
```

默认每个“类别×标签”抽取5题，目标共30题，seed为42。输出目录由脚本生成；已有同名结果可能来自早期代码版本，运行前应核对配置和实际唯一ID数。

### 4. 模糊鲁棒性实验

```bash
python run_robustness.py
```

当前配置为六个分层各15题，共90题；seed=42；半径为0、1、3、5；每次生成最多256个token。脚本尝试跳过已有成功记录，并离线重新解析回答。

**该入口保留历史逻辑，运行前需要处理下方列出的基线混用问题。它可能触发最多360次推理，不建议在不检查输出目录的情况下直接重跑。** 模型生成参数定义于 `models/qwen_vl.py`，多数实验参数直接定义于脚本中，当前没有命令行参数接口。

### 5. 分析已有实验

```bash
python analyze_robustness.py
```

该脚本读取 `results/pope90_v2_qwen3vl_blur{0,1,3,5}_seed42/results_parser_v2.jsonl`，打印指标、预测转移和半径5的 Yes→No 案例。结果文件不随仓库上传，因此新克隆的仓库需要先生成或自行恢复对应文件。

## 历史实验与已知问题

以下是2026-10-26对本机原始实验的核查记录，不是仓库内新运行的成绩，也不代表全量POPE基准。

- `pope90_v2_*` 四组各有90条唯一记录，45正45负，各类别各30题；旧 `pope90_*` 目录实际各87条，不能混用。
- `run_robustness.py` 的推理输出使用 `pope90_v2_*`，但后续基线路径及 `evaluators/robustness_summary.py` 仍引用旧 `pope90_*`。新实验应统一目录和配对清单，不能只比较目录名称。
- 当前 `evaluators/metrics.py` 的 Recall 是 `TP/(TP+FN)`，跳过了无效正样本，应理解为“有效回答条件召回率”。完整正样本召回率应使用 `TP/正标签总数`，并单独报告无效回答数与覆盖率。
- 当前 parser 可以漏掉长回答末尾的明确 Yes。历史90题四组的漏解析数分别为2、13、14、15；不能将这些全部视为视觉识别失败。
- 当前翻转率只在两边均有效的回答上计算；应同时报告分母、有效→无效转换和全体样本上的变化率。
- `pope_行号` 只在固定数据顺序内唯一；数据重排后可能变化。跨类别的 `question_id` 也会重复，后续应冻结完整样本清单并加入图像/问题哈希。
- 当前配置没有完整记录历史模型revision、生成随机性和包版本。断点续跑也不能处理截断JSONL末行等情况。

因此，仓库中的历史逻辑适合学习流程，正式实验还需完善身份校验、配置冻结、解析规则和配对统计。审计副本的修复结果没有作为本仓库已实现的功能宣传。

## 版本管理范围

上传代码、说明和依赖清单；本机数据、模型权重、实验输出、模糊示例图、IDE设置和缓存由 `.gitignore` 排除，原文件仍保留在本机。克隆仓库不包含数据和已跑结果。

## 参考

- [POPE论文](https://arxiv.org/abs/2305.10355)
- [POPE官方项目](https://github.com/RUCAIBox/POPE)
- [oplex/POPE 数据集](https://huggingface.co/datasets/oplex/POPE)
- [Qwen3-VL-2B-Instruct-4bit 模型](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)

数据和模型的使用、再分发请遵循各自来源的许可。
