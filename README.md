# HeartSound: 心音杂音三分类与多模态建模基准与实现

HeartSound 项目旨在构建一个基于 PhysioNet Challenge 2016 与 CirCor Heart Sound 2022 的统一评估基准，并实现面向心音杂音三分类（存在 / 不存在 / 不确定）的模型与训练评估流程。项目包含数据准备、增强与特征提取、模型训练与推理、不确定性与校准评估、网格实验与离线缓存/代理加速等完整工程链路。

- 任务：心音杂音三分类（Present / Absent / Unknown）
- 数据：PhysioNet 2016、CirCor Heart Sound 2022（病人级 7:2:1 划分）
- 模型：Heart-MambaFormer（自监督音频骨干 + 多尺度状态空间建模 + 跨位置注意力 + 元数据融合）
- 指标：macro-F1、weighted accuracy、AUROC、AUPRC（可扩展 ECE/MCE 校准指标）

---

## 目录
- 功能亮点
- 目录结构
- 环境与依赖
- 数据集准备
- 划分与元数据生成
- 训练与评估（含代理/离线）
- 配置说明
- 日志与产物
- 不确定性与 TTA（规划）
- 常见问题与排错
- 参考与致谢

---

## 功能亮点
- 自监督音频骨干（可接入 BEATs/AST/WavLM 等）与多尺度 SSM（Mamba 系）结合，线性复杂度建模长序列。
- 多听诊位置多实例汇聚（跨位置注意力），融合人口学元数据（Age/Sex/Height/Weight/Pregnancy）。
- 数据增强：带通（20–800 Hz）、增益扰动、带阻、加噪（可控 SNR）、SpecAugment。
- 工程化训练：Accelerate 混合精度、余弦退火 + warmup、分阶段微调、单机多卡/多实验网格脚本。
- 离线缓存与网络代理脚本，支持受限环境。

---

## 目录结构
```text
HeartSound/
  configs/                 # 训练配置（多个实验变体）
  docs/                    # 方法论与网络环境说明
  hf_cache/                # HuggingFace 缓存（已在 .gitignore 中忽略）
  logs/                    # 训练日志（忽略提交）
  metadata/                # 合并后的元数据（combined_metadata.*，忽略提交）
  runs/                    # 训练产物（checkpoints、history 等，忽略提交）
  scripts/                 # 数据切分、网格与代理训练脚本
  src/                     # 数据与模型代码
  train.py                 # 训练主脚本
  README.md
```

---

## 环境与依赖
推荐使用 Conda 环境（本仓库示例环境名：`heartsound-2025`）。

```bash
# 1) 启动 shell 并激活 conda（按你的系统路径调整）
source /map-vepfs/miniconda3/etc/profile.d/conda.sh
conda activate heartsound-2025

# 2) 安装核心依赖（示例，按需增减）
pip install -U pip
pip install torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install accelerate scikit-learn pyyaml tqdm
# 如需自监督骨干（Hugging Face）：
pip install transformers huggingface_hub
```

> 说明：请根据你的 CUDA/驱动选择合适的 PyTorch 轮子；若在离线/受限环境，参见下文“代理与离线缓存”。

---

## 数据集准备
建议将数据放在独立目录（不纳入 Git）：

- 机器路径（示例，见 `DATASET_Q&A.md`）：`/map-vepfs/qinyu/CodeSpace/datasets`
- CirCor 2022 根目录：`/map-vepfs/qinyu/CodeSpace/datasets/physionet-circor-heart-sound/files/circor-heart-sound/1.0.3`
- PhysioNet 2016 根目录：`/map-vepfs/qinyu/CodeSpace/datasets/physionet-challenge-2016/files/challenge-2016/1.0.0`

> 请确保音频（.wav）与官方标注/表格文件在各自数据集的默认结构中。

---

## 划分与元数据生成
项目统一在“病人级”构建 7:2:1 的 train/val/test 划分，并将两个数据集合并为统一元数据清单。

```bash
source /map-vepfs/miniconda3/etc/profile.d/conda.sh
conda activate heartsound-2025

python scripts/create_splits.py \
  --circor-root /map-vepfs/qinyu/CodeSpace/datasets/physionet-circor-heart-sound/files/circor-heart-sound/1.0.3 \
  --physionet2016-root /map-vepfs/qinyu/CodeSpace/datasets/physionet-challenge-2016/files/challenge-2016/1.0.0 \
  --out-dir metadata \
  --seed 2025
```

输出文件（默认）：`metadata/combined_metadata.json` 与 `metadata/combined_metadata.csv`（已在 `.gitignore` 中忽略）。

---

## 训练与评估

### 方式 A：直接运行训练脚本
```bash
source /map-vepfs/miniconda3/etc/profile.d/conda.sh
conda activate heartsound-2025

python train.py \
  --config configs/heart_mambaformer.yaml \
  --output runs
```
- 指标：训练会在验证集上计算 macro-F1、weighted accuracy、AUROC、AUPRC，并在测试集上报告最终指标。
- 产物：`runs/<experiment_name>/`（包含 `best.pt`, `model.safetensors`, `history.json` 等）。

### 方式 B：代理与缓存环境（推荐在受限网络下）
```bash
bash scripts/train_with_proxy.sh \
  --config configs/heart_mambaformer_small.yaml \
  --gpus 0 \
  --output runs
```
- 脚本会：设置 HTTP 代理、配置 HF 缓存到 `hf_cache/`、选择 GPU 并启动训练。

### 方式 C：多实验网格启动（单机多卡并行）
```bash
bash scripts/launch_grid.sh
```
- 自动在 0–7 号 GPU 上并行启动配置于 `configs/exp*.yaml` 的 8 个实验，日志在 `logs/`。

---

## 配置说明（示例字段）
以 `configs/heart_mambaformer.yaml` 为例：

```yaml
experiment_name: heart_mambaformer_v1
metadata_path: metadata/combined_metadata.json
sample_rate: 4000
max_duration: 20.0
max_locations: 4

model:
  # HeartMambaFormer 的超参数（如隐藏维度、SSM 块数、元数据嵌入等）
  hidden_size: 256
  num_layers: 8
  metadata_dim: 64

train:
  batch_size: 16
  num_epochs: 50
  learning_rate: 3.0e-4
  weight_decay: 0.01
  mixed_precision: bf16   # 可选：no/fp16/bf16
  warmup_epochs: 2
  gradient_accumulation_steps: 1

scheduler:
  min_lr: 1.0e-6

logging:
  output_dir: runs
```

> 你可以参考 `configs/exp*.yaml` 进行变体实验（如解冻策略、损失函数切换、对比学习温度搜索等）。

---

## 日志与产物
- 日志：`logs/*.log`（由脚本自动写入，已在 `.gitignore` 中忽略）。
- 训练产物：`runs/<experiment>/`（权重与训练历史，忽略提交）。
- Hugging Face 缓存：`hf_cache/`（避免默认 `~/.cache` 空间问题）。

---

## 不确定性与 TTA（规划）
- 推理时的 Test-Time Augmentation（TTA）：不同增益/遮挡种子，输出合并均值与方差。
- 温度标定与校准指标：ECE/MCE 统计与温度缩放，纳入验证流程。

> 初版脚本已包含主要训练指标与日志记录，不确定性与校准的评估将于后续 PR 中补齐。

---

## 常见问题与排错
- Q：拉取骨干权重失败 / 下载慢？
  - A：使用 `scripts/train_with_proxy.sh`，或设置 `HF_HOME`/`TRANSFORMERS_CACHE` 到本地 `hf_cache/` 后重试。
- Q：CUDA 内存碎片或 OOM？
  - A：降低 `batch_size`，或设置 `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`；减少模型深度/宽度。
- Q：无法访问外网（完全离线）？
  - A：提前将所需模型权重拷贝进 `hf_cache/`，并设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。
- Q：数据路径不一致？
  - A：请在 `scripts/create_splits.py` 中通过命令行参数覆盖数据根目录；或在配置中指定 `metadata_path`。

---

## 参考与致谢
- 数据集：
  - PhysioNet/CinC Challenge 2016（心音正常/异常/不确定）
  - CirCor DigiScope / PhysioNet Challenge 2022（多位置、细粒度杂音注释）
- 自监督与骨干：BEATs/AST/WavLM/Audio-Mamba 等
- 状态空间模型：Mamba 及相关选择性扫描机制

---

## 贡献与许可证
- 欢迎提交 Issue 或 PR，一起完善周期分割、SSM/LCAP 细节与不确定性评估。
- 许可证：若未另行声明，默认为研究用途优先（可按需在仓库根目录添加 `LICENSE`）。

---

## 快速开始（TL;DR）
```bash
# 0) Conda 环境
source /map-vepfs/miniconda3/etc/profile.d/conda.sh
conda activate heartsound-2025

# 1) 生成元数据
python scripts/create_splits.py \
  --circor-root /map-vepfs/qinyu/CodeSpace/datasets/physionet-circor-heart-sound/files/circor-heart-sound/1.0.3 \
  --physionet2016-root /map-vepfs/qinyu/CodeSpace/datasets/physionet-challenge-2016/files/challenge-2016/1.0.0 \
  --out-dir metadata

# 2) 训练
python train.py --config configs/heart_mambaformer.yaml --output runs

# 3) 查看日志与结果
ls logs/ && ls runs/
```

如需帮助或协作，请在仓库中开 Issue。祝研究顺利！
