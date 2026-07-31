# MADAv2
Project for [MADAv2: Advanced Multi-Anchor Based Active Domain Adaptation Segmentation](https://arxiv.org/abs/2301.07354) (accepted by TPAMI), which is modified from [Multi-Anchor Active Domain Adaptation for Semantic Segmentation](https://arxiv.org/abs/2108.08012) (ICCV Oral 2021).

> **Abstract.**
> Unsupervised domain adaption has been widely adopted in tasks with scarce annotated data.
> Unfortunately, mapping the target-domain distribution to the source-domain unconditionally may distort the essential structural information of the target-domain data, leading to inferior performance.
To address this issue, we firstly propose to introduce active sample selection to assist domain adaptation regarding the semantic segmentation task.
> By innovatively adopting multiple anchors instead of a single centroid, both source and target domains can be better characterized as multimodal distributions, in which way more complementary and informative samples are selected from the target domain.
> With only a little workload to manually annotate these active samples, the distortion of the target-domain distribution can be effectively alleviated, achieving a large performance gain.
> In addition, a powerful semi-supervised domain adaptation strategy is proposed to alleviate the long-tail distribution problem and further improve the segmentation performance.
> Extensive experiments are conducted on public datasets, and the results demonstrate that the proposed approach outperforms state-of-the-art methods by large margins and achieves similar performance to the fully-supervised upperbound, *i.e.*, 71.4\% mIoU on GTA5 and 71.8\% mIoU on SYNTHIA.
> The effectiveness of each component is also verified by thorough ablation studies. 

![](./img/visualization.png)
As shown in the figure, our features are perfectly distributed around the target centers, while traditional features of adversarial training tend to deviate from the real target distribution.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Notes](#notes)

## Requirements

The code requires Pytorch >= 0.4.1 and faiss-cpu >= 1.7.2. The code is trained using a NVIDIA RTX3090 with 24GB memory.

## Usage

### Controlled image-level reproduction

The `scripts/` and `tools/` additions provide a portable reproduction path
for the original DeepLab-101 model and for DINOv3-B + ReIN + HRDA. The latter
is self-contained under `models/vfm/` and does not import the SSDA, RIPU,
HALO, or D2ADA repositories at runtime.

The controlled protocol uses exactly the same image-level budgets as the
TC-SSDA comparison:

| Setting | Target pool | Images queried |
| --- | ---: | ---: |
| GTA5 to Cityscapes | 2,975 | 47 (1/64) |
| SYNTHIA to Cityscapes | 2,975 | 47 (1/64) |
| Cityscapes to ACDC | 1,600 | 25 (1/64) |
| Cityscapes to MUSES | 1,500 | 24 (1/64) |
| Cityscapes to Mapillary | 18,000 | 141 (1/128) |

`MAX_ITERS=40000` is the total adaptation budget. By default it is divided
equally between MADAv2 stage 1 and stage 2, rather than assigning 40k
iterations to each stage. Validation and checkpointing occur every 10k local
stage iterations, and training logs are printed every 100 iterations.

#### Environment and assets

The recommended environment is the existing `reinpy10` environment:

```bash
DATA_ROOT=/path/to/datasets \
PRETRAINED_ROOT=/path/to/pretrained \
CONDA_ENV=reinpy10 \
bash scripts/setup_env.sh
```

`DATA_ROOT` should contain `gta`, `synthia`, `cityscapes`, `acdc`, `muses`,
and `mapillary`. Individual paths such as `GTA_ROOT=/path/to/GTAV` override
that convention. `PRETRAINED_ROOT` should contain
`resnet/resnet101-5d3b4d8f.pth` and `dinov3/dinov3_vitb16.pth`; the explicit
`RESNET101_PRETRAINED` and `DINOV3_PRETRAINED` variables are also accepted.

The setup script checks the Python dependencies, creates direct raw-dataset
links under `data/`, copies both pretrained checkpoints into this repository,
and verifies the repository-owned split lists. FAISS is optional: anchor
clustering automatically falls back to scikit-learn `MiniBatchKMeans`.

To create an isolated clone of `reinpy10` instead:

```bash
DATA_ROOT=/path/to/datasets \
PRETRAINED_ROOT=/path/to/pretrained \
CONDA_ENV=reinpy10 \
CREATE_ENV=1 TARGET_ENV=madav2 \
bash scripts/setup_env.sh
```

Before a long server run, verify both segmentors with real data:

```bash
GPU=0 CONDA_ENV=reinpy10 bash scripts/smoke_test.sh
```

This performs one optimization step, checkpoint save/reload, and two-image
validation for DeepLabV3+ and DINOv3-B+ReIN+HRDA. Set `MODELS=deeplab101` or
`MODELS=dinov3_base_rein_hrda` to test only one path.

#### One-command training

Run either model by changing `MODEL`:

```bash
DATASET=gta2cityscapes \
MODEL=dinov3_base_rein_hrda \
GPU=0 \
CONDA_ENV=reinpy10 \
bash scripts/run_experiment.sh
```

The supported dataset names are:

```text
gta2cityscapes
synthia2cityscapes
cityscapes2acdc
cityscapes2muses
cityscapes2mapillary
```

Use `MODEL=deeplab101` for the official segmentation architecture. A detached
server launch can be written as:

```bash
screen -dmS madav2_g2c bash -lc \
  'cd /path/to/MADAv2 && DATASET=gta2cityscapes MODEL=dinov3_base_rein_hrda GPU=0 CONDA_ENV=reinpy10 bash scripts/run_experiment.sh'
```

For the original GTA5-to-Cityscapes schedule (5% images, 90k source warm-up,
and 900k iterations in each adaptation stage), override the controlled
defaults explicitly:

```bash
DATASET=gta2cityscapes MODEL=deeplab101 \
BUDGET=149 RATIO_NAME=5pct \
SOURCE_ITERS=90000 MAX_ITERS=1800000 STAGE1_ITERS=900000 \
bash scripts/run_experiment.sh
```

The pipeline trains a source-only initialization, extracts source/target
features, clusters ten anchors per domain, records the exact selected image
list, trains stage 1, refreshes target anchors, and trains stage 2. Completed
source, selection, stage-1, and stage-2 artifacts are skipped on rerun.
The launcher caps BLAS/OpenMP libraries at eight CPU threads to avoid
OpenBLAS metadata overflows on high-core-count servers. Override this safe
default with `MADAV2_BLAS_THREADS=<1--64>` when needed.

Each experiment is saved as:

```text
runs/controlled/<setting>/<ratio>/<model>/
  source/
  selection/
    selected_images.txt
    selected_indices.txt
    selection_metadata.json
    acquisition_timing.json
    stage2_preparation_timing.json
  stage1/
  stage2/
```

`acquisition_timing.json` separates source feature extraction, source
clustering, target feature extraction, target clustering, and final image
selection. Post-stage-1 anchor refresh is recorded separately and is not
counted as query time.

Evaluate every saved checkpoint (or pass selected filenames after
`--checkpoints`):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n reinpy10 \
python tools/evaluate_checkpoints.py \
  --config runs/controlled/gta2cityscapes/1_64/dinov3_base_rein_hrda/stage2.yml \
  --run-dir runs/controlled/gta2cityscapes/1_64/dinov3_base_rein_hrda/stage2
```

1. Preparation:
* Download the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset as the source domain, and the [Cityscapes](https://www.cityscapes-dataset.com/) dataset as the target domain.
* Download the [Weights](https://drive.google.com/drive/folders/1Ln-fTBTivmMGJdRiVOi1774eBK_GMrhZ?usp=sharing) and [Features](https://drive.google.com/drive/folders/17DMUHU97X5JPnEi9Hx8xWv-YYRDKdfie?usp=sharing). Move features to the MADAv2 directory.

2. Set up the config files.
* Set the data paths
* Set the pretrained model paths

3. Quickstart
* To run the code with our weights and anchors:
~~~~
python3 step1_train_active_sup_only.py
python3 step2_train_active_semi_sup.py
~~~~
* During the training, the generated files (log file) will be written in the folder 'runs/..'.

4. Evaluation
* Set the config file for test (configs/test_from_city_to_gta.yml):
* Run test.py to see the results:
~~~~
python3 test.py
~~~~

5. Training-whole process
* Setting the config files.
* Stage 1:
* 1-Save the features for source and target domains with the warmup model:
~~~~
python3 step1_save_feat_source.py
python3 step1_save_feat_target_warmup.py
~~~~
* 2-Cluster the features of source and target domains:
~~~~
python3 step1_cluster_anchors_source.py
python3 step1_cluster_anchors_target_warmup.py
~~~~
* 3-Select the active samples by considering the distance from the both domains:
~~~~
python3 step1_select_active_samples.py
~~~~
* 4-Training with the active samples:
~~~~
python3 step1_train_active_sup_only.py
~~~~

* Stage 2:
* 1-Save the features of target samples with the stage1 model:
~~~~
python3 step2_save_feat_target.py
~~~~
* 2-Cluster the features of target samples:
~~~~
python3 step2_cluster_anchors_target.py
~~~~
* 3-Training with the proposed semi-supervised domain adaptation strategy: 
~~~~
python3 step2_train_active_semi_sup.py
~~~~



## License

[MIT](LICENSE)

The code is heavily borrowed from the CAG_UDA (https://github.com/RogerZhangzz/CAG_UDA) and U2PL (https://github.com/Haochen-Wang409/U2PL).

If you use this code and find it usefule, please cite:
~~~~
@article{ning2023madav2,
  title={MADAv2: Advanced Multi-Anchor Based Active Domain Adaptation Segmentation},
  author={Ning, Munan and Lu, Donghuan and Xie, Yujia and Chen, Dongdong and Wei, Dong and Zheng, Yefeng and Tian, Yonghong and Yan, Shuicheng and Yuan, Li},
  journal={arXiv preprint arXiv:2301.07354},
  year={2023}
}
~~~~

## Notes
We also provide the results of D2ADA version in [Weights_D2ADA](https://drive.google.com/drive/folders/1pnSJZ-WWkYivdRokD9rteyPQ4DVWeGcu?usp=sharing).

As you see, our framework is kind of out of date. If you want to continue in the research of domain adaptation, we recommend you to use the [D2ADA](https://github.com/tsunghan-wu/D2ADA) framework, which is more powerful and easy to use.
