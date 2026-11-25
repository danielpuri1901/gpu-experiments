# Classical models, vision tasks and parallel computing, 2026-11-25

One full run of `models-classical-vision-parallel.ipynb` (commit `367b52c`) in an interactive Kaggle T4 x2 session, saved as Version 3 with its outputs. All 18 parts passed in 9.8 minutes.
Version 2 gave the same numbers except for OCR. Its OCR grader compared letter case against a model that writes upper case. The grader now ignores case.
One seed. Intervals are 95% Wilson. Paired tests are exact McNemar.

## When does a neural network win?

Tabular data (Covertype, 7 classes, always-majority baseline 48.9%):

| Training rows | Logistic regression | Random forest | Gradient boosting | Neural network (MLP) |
|---|---:|---:|---:|---:|
| 1,000 | 70.1 | **71.5** | 70.5 | 69.3 |
| 10,000 | 71.7 | **81.0** | 78.4 | 76.4 |
| 100,000 | 72.1 | **90.8** | 81.3 | 86.1 |

Images (CIFAR-10, baseline 9.8%):

| Training images | Logistic regression on pixels | HOG and linear SVM | CNN |
|---|---:|---:|---:|
| 1,000 | 27.2 | 39.3 | 50.8 |
| 10,000 | 37.3 | 47.8 | 68.6 |
| 50,000 | 41.0 | 49.8 | 73.6 |

- On tables the forest wins at every size. The network passes gradient boosting only at 100,000 rows.
- On images the CNN wins at every size, and the gap grows with data (hand-made HOG features: 39 to 50%).
- One prediction costs 0.6 ms for logistic regression and the MLP, 23 ms for the forest and 8 ms for boosting.

## What each vision task takes in and gives back

| Task | Model | Score |
|---|---|---|
| Classification (one label) | CLIP ViT-B/32, zero-shot | 89.3% of 5,000 |
| Detection (boxes) | Mask R-CNN ResNet-50 | AP50 0.64, mAP 0.43 (200 images) |
| Instance segmentation (masks) | same | AP50 0.60, mAP 0.40 |
| Semantic segmentation (one class per pixel) | LRASPP MobileNetV3 | mean IoU 0.63, person IoU 0.79 (an all-background answer scores 0.04) |
| OCR | TrOCR | character error rate 0.015 clean, 0.59 at noise 0.1, 0.96 at noise 0.25 (empty reply: 1.0) |

- Semantic segmentation cannot count people that touch. On 61 images with 2 or more people, blobs were off by 2.07 people on average and instances by 0.93.
- OCR breaks before the image looks broken. Exact text was right for 41 of 48 clean lines and 0 of 48 at noise 0.1. The model then invents fluent wrong words ("inventaquint codeson").

## Why do GPUs help?

| | Result |
|---|---|
| GPU beats all CPU threads in matrix multiply from | n = 128 (241 against 129 GFLOPS) |
| Same, with the copy to the GPU counted, from | n = 512 |
| GPU fp32 and fp16 at n = 8,192 | 3,184 and 42,346 GFLOPS (CPU 265) |
| Throughput, batch 1 to 4,096 | 44x on the GPU. GPU over CPU: 1.7x at batch 1, 50x at batch 4,096 |
| Amdahl fit, serial fraction | 0.46 at n = 128, 0.35 at n = 2,048 (4 shared threads, so a rough number) |
| Two GPUs, strong scaling (global batch 256) | 2.37x, 119% efficiency |
| Two GPUs, weak scaling (256 per GPU) | 1.93x, 96% efficiency |

Strong scaling is above 100% because a batch of 128 per GPU runs more efficiently on this model than 256. An all-reduce takes 11 ms, and it overlaps with the backward pass.

## Robustness and shortcuts

| Test on the CNN | Accuracy | Calibration error (ECE) |
|---|---:|---:|
| Clean | 73.6 | 0.03 |
| Noise 0.04, 0.16 | 59.4, 25.2 | 0.08, 0.34 |
| Blur 1.0, 1.5 | 38.5, 28.0 | 0.17, 0.27 |
| Brightness +0.25, +0.5 | 66.6, 38.7 | 0.04, 0.09 |
| FGSM 1, 4 of 255 | 40.7, 3.6 | 0.24, 0.68 |

A high accuracy on the clean set says little about shifted data, and confidence stays high while accuracy falls.

| Planted shortcut | With the mark as in training | Mark removed |
|---|---:|---:|
| Image: magenta square on one class | 75.4 | 65.5 (12.4 when the mark sits on the other classes) |
| Tabular: a column that copies the label | 95.2 | 59.5 (baseline 49.4) |

In the Version 2 run (same code for these parts), Grad-CAM put 9.5% of its mass on the mark, which covers 1.6% of the image, and permutation importance ranked the mark 1 of 55 columns. The image test shows the model saw the mark and not the object: recall of the marked class fell from 99.6% to 0%.

## Foundation model against task model

| Labels per class | Task-specific CNN | CLIP zero-shot (no labels) |
|---|---:|---:|
| 10 | 33.5 | 89.3 |
| 100 | 50.8 | 89.3 |
| 1,000 | 68.6 | 89.3 |

CLIP wins against every CNN in the table (exact McNemar p below 0.001).

- Embeddings put similar images near each other. Nearest neighbour on CLIP embeddings scored 88.6%, on raw pixels 28.3%.
- Image and text retrieval on 500 queries: R@5 96.6% for dense search and 97.0% for hybrid with a keyword weight of 0.25. The difference is not measurable.
- A mixture of experts (8 experts, top 2) holds 8 times the parameters of the dense layer, uses 2 times as many for each token, and took 2.7 times as long per 4,096 tokens in this run.

## What the numbers show

- **Pick the model from the data type.** Trees for tables, networks for images, a foundation model when labels are few.
- **Small batches waste a GPU.** A batch of 1 gives 1.7x over the CPU, and 4,096 gives 50x.
- **Test on shifted data before you trust accuracy.** Clean 74%, FGSM 4%.
- **A shortcut looks like skill.** Remove the suspected feature and measure the drop.
- **Read the output type of a vision task first.** A semantic map cannot count objects.

## Limits

One run and one seed per row, 200 images in parts B2 and B3, 48 lines in OCR. The CNN has no augmentation, on purpose. Timings depend on the Kaggle machine.
