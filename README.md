# Epitope prediction

AsEP 表位预测模型的训练与评估代码。GitHub 只同步代码、配置和测试；数据、缓存、checkpoints 和结果保存在各台机器本地。

```text
baselines/             模型、数据加载和统一训练入口
configs/               14 个模型/消融变体的配置
tests/                 入口与路径配置测试；模型测试位于对应模型目录
scripts/               环境/数据检查、提交文件检查
machine.example.json   本机路径配置示例
```

## 安装与数据

在仓库根目录运行，使用 Python 3.10：

```bash
conda env create -f environment.yml
conda activate epitope-se3bench
# 根据目标 GPU 和驱动选择 wheel；下面是原训练环境的 CUDA 12.8 版本。
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
cp machine.example.json machine.local.json
```

其他 CUDA wheel 见 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v280)。无需安装完整上游工程或 `torch_geometric`。

编辑 `machine.local.json` 的 `data_root`，指向单独准备的 AsEP 数据目录：

```text
asepv1.2.0/
├── asepv1-AbDb-IDs.txt
├── split_dict.pt
├── asepv1_interim_graphs/<id>.pt
└── structures/<id>.pdb
```

IgFold/ESM2 特征已包含在 interim graphs 中。各机器使用同一份数据和划分文件，缓存在本机重新构建。默认 `cache_root=.local/cache`、`runs_root=.local/runs`，也可改为仓库外的绝对路径；相对路径从仓库根目录解析。`machine.local.json` 和 `.local/` 已被 Git 忽略。

```bash
python scripts/doctor.py --check-data --require-cuda
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q
```

可用 `EPITOPE_DATA_ROOT`、`EPITOPE_CACHE_ROOT`、`EPITOPE_RUNS_ROOT` 覆盖本机配置。`doctor` 会显示 IDs 和 split 的 SHA-256，便于各机器核对。

## 训练与评估

```bash
# 查看实际参数，不访问数据、不训练。
python -m baselines.run --config configs/lf3drope.yaml --run-name lf_seed42 --dry-run

# 首次运行先准备缓存；多进程训练共用缓存时，先完成这一步。
python -m baselines.run --config configs/lf3drope.yaml --run-name prepare --build-cache-only

# 每个实验使用独立名称。
CUDA_VISIBLE_DEVICES=0 python -u -m baselines.run \
  --config configs/lf3drope.yaml --run-name lf_seed42

# 从原实验的实际配置重新评估最佳 checkpoint。
python -m baselines.run --config .local/runs/lf_seed42/config.resolved.json --eval-only
```

DeepInteract 需要完整链上下文缓存，训练前用 `configs/deepinteract.yaml` 单独执行一次 `--build-cache-only`。旧缓存指纹不匹配时，为普通缓存准备命令追加一次 `--set dataset.rebuild_cache=true`；DeepInteract 上下文缓存可更换 `dataset.deepinteract_context_cache` 文件名后重建。

默认使用 GPU；CPU 调试须加 `--set try_gpu=false`。新机器可用独立实验名配合 `--set hparams.max_epochs=1 --set debug_limit=4` 做流程检查。小样本检查结果不用于正式统计。

每个实验保存 `config.resolved.json`、`run_metadata.json`、`best_model.pt`、`history.jsonl`、`final_metrics.json/.pt`，包含参数、Git 版本、机器/环境和数据划分校验值。训练拒绝覆盖非空目录；重新训练请换新名字。`--eval-only` 会重新写入该实验的最终指标。

## 配置与模型迭代

`configs/` 支持 `lf3drope`、`gf3drope`、`gf3drope_rotaug`、`1drope`、`distance_rbf`、`local_geometry`、`ipa`、`gvp`、`geformer`、`deepinteract`、`lf3drope_sparse_edge`、`lf3drope_geometry_value`、`lf3drope_bond_angle` 和 `lf3drope_module_ablation`。

复制一个配置并修改后提交 Git，或通过 `--set` 覆盖参数：

```bash
python -m baselines.run --config configs/lf3drope.yaml --run-name lf_ratio_seed43 \
  --set seed=43 --set split=ratio --set dataset.split_method=epitope_ratio
```

`bits` 是 PLM/aux/CDR/jitter 四字符字符串，须与 embedding、aux、CDR 和 augmentation 设置一致；更换特征组合时使用不同缓存文件。增大模型可调整 `hparams.d_model`、`dim_ff`、层数等。模块消融使用 `hparams.use_cross_attention/use_global_context/use_pair_mil`。

新模型放入 `baselines/<name>/model.py`，提供兼容的 `Model` 接口，在 `baselines/run.py` 的 `METHODS` 注册并添加配置；需要特殊输入时接入 `configured_trainer`。现有训练器每步处理一个复合物，不支持 DDP、自动增大 batch size、AMP 或断点续训。可在大显存 GPU 上运行更大模型或多个独立实验。

## 同步与汇总

提交前运行 `python scripts/check_repository.py`，确认没有数据、结果或大文件混入。训练期间保持代码目录不变，结束后再 `git pull --ff-only`；同时开发可使用独立 clone/worktree。

人工汇总时按“机器名/实验名”复制 `final_metrics.json`、`config.resolved.json` 和 `run_metadata.json`，需要学习曲线时再复制 `history.jsonl`。比较时保持数据版本与实验设置一致；缓存和旧实验指纹包含本机路径，不能仅凭它们跨机判等。模型选择使用验证集，测试集用于最终报告。

模型来源、适配差异和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
