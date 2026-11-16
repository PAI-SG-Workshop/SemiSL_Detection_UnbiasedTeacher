# SemiSL_Detection_UnbiasedTeacher

**Semi-supervised CenterNet (Unbiased-Teacher style) on FashionLetters36 - PAI-SG SemiSL**

Part of the *Physical AI* reference series of the **Statera Working Group**. Semi-supervised detection part of
[SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36); SSL backbones from [SSL_Redundancy_BarlowTwins](https://github.com/Statera-Workshop/SSL_Redundancy_BarlowTwins).

## Method (`code/semisl_det_v2.py`)

Unbiased Teacher (Liu et al. 2021) adapted to the anchor-free CenterNet of
[OD_Image_1-Stage_CenterNet](https://github.com/Statera-Workshop/OD_Image_1-Stage_CenterNet) v1.1: supervised burn-in on the labeled images,
then an EMA teacher (0.999) labels the weak view of unlabeled images (score >= 0.3, common NMS) and the student
learns a heat-map focal loss on a photometric strong view of the same image (geometry unchanged). Background is
learned only where the teacher heat is < 0.1 (cells in between ignored), images without pseudo boxes are skipped,
the loss is normalised by max(1, pseudo boxes), size / offset losses are not used for pseudo boxes. Teacher BN uses
batch statistics during training; PreciseBN before evaluation. Same updates, protocol and test set as the
labeled-only CenterNet runs (the baseline). Labeled splits: nested image subsets, seed 777 (`splits/`).
Pseudo-label precision / recall against the unlabeled ground truth are logged, never trained on.

**Version 1 failed** (`code/semisl_det.py`, kept as a record): threshold 0.5 gave 0.1-2 pseudo boxes per image
(true ~6.5), every missed object became a background target, and mAP fell far below the labeled-only baseline.

## Results (teacher test mAP50)

| Labels | Init | labeled-only 30 ep | + SemiSL 30 ep | labeled-only 90 ep | + SemiSL 90 ep |
| --- | --- | --- | --- | --- | --- |
| 1 % | from scratch | 0.139 | 0.145 | 0.131 | - |
| 1 % | Barlow 200 | 0.319 | 0.319 | 0.330 | 0.348 |
| 5 % | from scratch | 0.463 | 0.414 | 0.605 | **0.725** |
| 5 % | Barlow 200 | 0.676 | 0.687 | 0.711 | **0.773** |
| 10 % | from scratch | 0.564 | 0.378 (burn-in 60 %: 0.571) | 0.725 / 0.720 | **0.793 / 0.801** |
| 10 % | Barlow 200 | 0.771 | 0.778 | 0.770 / 0.774 / 0.775 | **0.821** |

30 epochs: burn-in 30 % (60 % where noted); 90 epochs: burn-in 60 %. From-scratch CenterNet with 100 % labels:
0.611 (30 epochs), 0.862 / 0.859 (90 epochs). All runs with pseudo-label statistics: `results/det_summary.md`.

1. **The budget decides.** At 30 epochs the unlabeled term adds nothing on a good backbone and hurts with a weak
   teacher; at 90 epochs with burn-in 60 % it adds **+0.06 to +0.12**, reproduced with a second seed.
2. **SSL and semi-supervised learning add up**: Barlow 200 + SemiSL with 10 % labels 0.821 vs 0.722 for
   from-scratch with the same labels and 0.860 with 100 % labels.
3. **Pseudo-label precision is the limit**: with 1 % labels (41 images) the teacher stays weak (+0.018).

## Weights

Release `v1.0`: the PreciseBN teacher of the six 90-epoch runs (`det2_*_e90*_teacher.pth`, CenterNet state dicts
for `CenterNetSSL(36)` of `code/train_image_centernet_ssl.py`; SHA-256 in `SHA256SUMS.txt`).

## Run (from the OD_Image_1-Stage_CenterNet folder, with the files of `code/` copied next to its scripts)

```bash
python make_label_splits.py                            # ~/SSL_Wave2/roots/p01 ... p25
python semisl_det_v2.py --ratio p10 --init scratch --epochs 90 --burnin 0.6 --tag det2_p10_scratch_e90
python semisl_det_v2.py --ratio p10 --init <ssl_barlow_e200_backbone.pth> --epochs 90 --burnin 0.6 --tag det2_p10_barlow200_e90
PHYSICAL_AI_ROOT=~/SSL_Wave2/roots/p10 python train_image_centernet_lab.py --tag lab_p10_scratch_e90 --epochs 90   # baseline
```

Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md).
