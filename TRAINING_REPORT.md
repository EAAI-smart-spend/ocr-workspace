# OCR Training Report (Academic Summary)

This document summarizes the OCR training experiments conducted in this repository, focusing on Chinese (Traditional) receipt text recognition. The goal is to develop a receipt-robust recognizer and evaluate the practical trade-offs between (1) fine-tuning an existing EasyOCR recognizer and (2) training a recognizer from scratch using the *deep-text-recognition-benchmark* (DTRB) framework.

## 1. Problem Statement

Receipt OCR is challenging because of:
- Mixed scripts and symbols (Chinese/English, currency, punctuation)
- Diverse fonts, sizes, and printing quality
- Noise from photo capture (blur, skew, lighting)

The target setting in this workspace is **line-level text recognition** (cropped line images paired with a text label), trained and evaluated under DTRB.

## 2. Datasets

### 2.1 Real receipt dataset (line-level)

We use a real receipt line dataset organized under `dataset/real_receipt/`.

From training logs (DTRB dataset summary):
- **Train**: 2126 samples
- **Validation**: 265 samples

This dataset is expected to better match the target domain than synthetic-only data.

### 2.2 Synthetic data (earlier exploration)

Earlier exploration included synthetic TRDG/HF generation (EN+ZH). Empirically, models can reach near-zero training loss quickly while validation accuracy stays low, suggesting **distribution mismatch** between synthetic training data and real receipt validation data.

## 3. Model and Training Framework

All experiments use DTRB-style modular recognizers:

- **Transformation**: TPS (Thin Plate Spline) for geometric normalization
- **FeatureExtraction**: VGG or RCNN
- **SequenceModeling**: BiLSTM
- **Prediction**: CTC

### 3.1 Two main training strategies

1. **Fine-tune from EasyOCR pre-trained weights** (`pre_trained_models/chinese.pth`)
2. **Train from scratch** (random initialization)

## 4. Experiments and Results

### 4.0 Summary table (best vs final)

Metrics are reported on the validation set. “Best” refers to the best validation accuracy observed in the available log excerpts for each run.

| Experiment | Strategy | Architecture | Iterations | Best (iter) | Final (iter) |
|---|---|---|---:|---|---|
| A | Fine-tune from `chinese.pth` | TPS + VGG + BiLSTM + CTC | 20000 | acc 18.113%, norm_ED 0.41 (6000) | acc 16.226%, norm_ED 0.40 (5000) |
| B | From scratch | TPS + RCNN + BiLSTM + CTC | 5000 | acc 58.868%, norm_ED 0.79 (5000) | acc 58.868%, norm_ED 0.79 (5000) |

### 4.1 Experiment A — Fine-tune EasyOCR model (TPS + VGG + BiLSTM + CTC)

**Goal**: Adapt EasyOCR `chinese.pth` to receipt lines with a conservative learning rate.

**Command (from `workspace_step3.ipynb`)**

```bash
python deep-text-recognition-benchmark/train.py \
  --train_data "dataset/real_receipt/train" \
  --valid_data "dataset/real_receipt/valid" \
  --workers 0 \
  --num_iter 20000 \
  --valInterval 500 \
  --saved_model "pre_trained_models/chinese.pth" \
  --FT \
  --adam \
  --lr 5e-5 \
  --select_data / \
  --batch_ratio 1 \
  --Transformation "TPS" \
  --FeatureExtraction "VGG" \
  --SequenceModeling "BiLSTM" \
  --Prediction "CTC" \
  --input_channel 1 \
  --output_channel 512 \
  --hidden_size 512
```

**Observed behavior (from `saved_models/TPS-VGG-BiLSTM-CTC-Seed1111-20251204-032439/log_train.txt`)**

- Loss decreases rapidly (training loss approaches near-zero).
- Validation metrics improve only modestly.

Selected checkpoints (validation):
- Iter 500: accuracy 10.566, norm_ED 0.36
- Iter 2000: accuracy 13.585, norm_ED 0.37
- Iter 5000: accuracy 16.226, norm_ED 0.40
- Iter 6000 (best seen in excerpt): accuracy 18.113, norm_ED 0.41

**Interpretation**

Fine-tuning converges quickly but exhibits limited generalization gains on the receipt validation set. This is consistent with earlier observations that pre-trained recognizers may not transfer well when the domain differs (receipt typography/layout, noise, mixed symbols).

**Recorded hyperparameters (from `saved_models/TPS-VGG-BiLSTM-CTC-Seed1111-20251204-032439/opt.txt`)**

- Data: `train_data=dataset/real_receipt/train`, `valid_data=dataset/real_receipt/valid`, `batch_size=192`, `workers=0`
- Schedule: `num_iter=20000`, `valInterval=500`, `grad_clip=5`
- Optimizer: Adam, `lr=5e-05`, `beta1=0.9`, `eps=1e-08`
- Model: `Transformation=TPS`, `FeatureExtraction=VGG`, `SequenceModeling=BiLSTM`, `Prediction=CTC`
- Input/width: `imgH=32`, `imgW=100`, `input_channel=1`, `output_channel=512`, `hidden_size=512`
- Text: `batch_max_length=25`, `sensitive=False`, `data_filtering_off=False`, `num_class=5381`

### 4.2 Experiment B — Train from scratch (TPS + RCNN + BiLSTM + CTC)

**Goal**: Train a recognizer specialized for receipts without relying on EasyOCR initialization.

**Command (from `workspace_step3.ipynb`)**

```bash
python deep-text-recognition-benchmark/train.py \
  --train_data "dataset/real_receipt/train" \
  --valid_data "dataset/real_receipt/valid" \
  --workers 0 \
  --num_iter 5000 \
  --valInterval 500 \
  --adam \
  --lr 1e-3 \
  --select_data / \
  --batch_ratio 1 \
  --Transformation "TPS" \
  --FeatureExtraction "RCNN" \
  --SequenceModeling "BiLSTM" \
  --Prediction "CTC" \
  --input_channel 1 \
  --output_channel 512 \
  --hidden_size 512
```

**Observed behavior (from `saved_models/TPS-RCNN-BiLSTM-CTC-Seed1111-20251204-030551/log_train.txt`)**

Validation improves substantially within a short schedule:
- Iter 500: accuracy 46.792, norm_ED 0.73
- Iter 1000: accuracy 51.698, norm_ED 0.76
- Iter 1500: accuracy 53.585, norm_ED 0.78
- Iter 3500: accuracy 56.226, norm_ED 0.79
- Iter 5000: accuracy 58.868, norm_ED 0.79

**Interpretation**

Training from scratch with RCNN features provides significantly better validation accuracy than fine-tuning in this dataset regime. This suggests that, for receipts, a domain-specialized model can outperform transferred weights, especially when pre-training data differs substantially.

**Recorded hyperparameters (from `saved_models/TPS-RCNN-BiLSTM-CTC-Seed1111-20251204-030551/opt.txt`)**

- Data: `train_data=dataset/real_receipt/train`, `valid_data=dataset/real_receipt/valid`, `batch_size=192`, `workers=0`
- Schedule: `num_iter=5000`, `valInterval=500`, `grad_clip=5`
- Optimizer: Adam, `lr=0.001`, `beta1=0.9`, `eps=1e-08`
- Model: `Transformation=TPS`, `FeatureExtraction=RCNN`, `SequenceModeling=BiLSTM`, `Prediction=CTC`
- Input/width: `imgH=32`, `imgW=100`, `input_channel=1`, `output_channel=512`, `hidden_size=512`
- Text: `batch_max_length=25`, `sensitive=False`, `data_filtering_off=False`, `num_class=5381`

## 5. Discussion: module choices, controlled comparisons, and failure modes

This section frames the observed results as controlled comparisons and links outcomes to specific architectural choices and data conditions.

### 5.1 Modular recognizer formulation

We follow the DTRB formulation of an OCR recognizer as a composition of four stages: **Transformation**, **FeatureExtraction**, **SequenceModeling**, and **Prediction**. Each stage contributes a distinct inductive bias and defines what variation the model can explain.

- **Transformation (TPS)**: Thin-Plate Spline rectification attempts to map the observed text line to a canonical geometry. This is intended to reduce nuisance variation from perspective skew, local warping (e.g., curved receipt paper), and imperfect crops.

- **FeatureExtraction (VGG or RCNN)**: The visual backbone converts the rectified image into a feature map that is later interpreted as a left-to-right sequence. In practice, this stage dominates the model’s capacity to represent receipt-specific appearance cues (thermal noise, low contrast, font idiosyncrasies).

- **SequenceModeling (BiLSTM)**: A bidirectional recurrent layer models contextual dependencies along the sequence dimension. This is beneficial when local glyph evidence is ambiguous and disambiguation requires neighboring characters (e.g., digits near currency symbols, similar Chinese radicals under noise).

- **Prediction (CTC)**: CTC provides alignment-free sequence supervision. Given line-level transcriptions (without character boxes), CTC is a natural objective because it marginalizes over frame-to-label alignments.

### 5.2 Controlled variable: FeatureExtraction backbone

To isolate the effect of the visual backbone, we keep `Transformation=TPS`, `SequenceModeling=BiLSTM`, and `Prediction=CTC` fixed, and vary only `FeatureExtraction`:

- Experiment A: `FeatureExtraction=VGG` with pre-trained initialization (`saved_model=pre_trained_models/chinese.pth`, `FT=True`).
- Experiment B: `FeatureExtraction=RCNN` with random initialization (`FT=False`).

Under the same dataset split (train=2126 lines, valid=265 lines), Experiment B achieves substantially higher validation performance (Table 4.0).

### 5.3 Evidence of limited transfer under domain shift

In Experiment A, training loss decreases rapidly, but validation accuracy remains low and unstable relative to the scratch RCNN baseline. For example, the best observed point in the log excerpt occurs at iteration 6000 (accuracy 18.113%, norm_ED 0.41), after which the validation signal fluctuates (e.g., iteration 7000 accuracy drops to 13.585%). This pattern is consistent with transfer under domain shift where the inherited representation does not match the target imaging/typography conditions.

### 5.4 Why fine-tuning underperformed (plausible mechanisms)

Based on the observed learning curves and the known differences between generic scene text and receipts, we attribute the weak transfer performance to the following interacting mechanisms:

1. **Train–test distribution mismatch**: Receipts exhibit unique nuisances (thermal printing artifacts, structured numeric fields, dense punctuation/currency patterns). A recognizer pre-trained on a broader scene-text distribution may allocate capacity to features that do not help in receipts, yielding limited positive transfer.

2. **Small real-data regime amplifies overfitting**: With ~2k labeled lines, a fine-tuned model can quickly memorize frequent templates while failing to generalize to rare tokens and unseen store layouts. Lowering the learning rate can slow parameter drift but does not resolve missing coverage.

3. **Backbone suitability and feature reuse**: If the pre-trained backbone’s intermediate features are not aligned with receipt-specific appearance cues (low contrast, thin strokes, repetitive digits), optimization can converge to a poor local optimum where loss is reduced primarily on frequent/easy patterns.

4. **Class-frequency skew in rich character sets**: The effective frequency of many symbols (rare punctuation forms, uncommon Chinese characters) is low. Even with `num_class=5381`, fine-tuning may not sufficiently update under-represented classes, reducing validation accuracy and edit-distance metrics.

### 5.5 Practical implication for deployment and reproducibility

The results suggest that, in this dataset regime, improving *domain match* and *backbone capacity for receipt artifacts* is more impactful than relying on cross-domain pre-training. If deployment requires EasyOCR integration, the recognizer architecture must exactly match the trained checkpoint configuration; otherwise, the weight loading step fails due to incompatible parameter shapes.

## 6. Conclusions

- **Fine-tuning EasyOCR (TPS+VGG+BiLSTM+CTC)** showed limited validation improvement on real receipt lines (best observed norm_ED ~0.41 in the provided log excerpt).
- **Training from scratch (TPS+RCNN+BiLSTM+CTC)** achieved substantially higher validation accuracy (best observed accuracy 58.868%, norm_ED 0.79 at 5000 iterations).
- The strongest practical takeaway is that **domain-matched real receipt line data** and an architecture suited to receipts (RCNN in this case) are more impactful than relying on pre-trained weights from a different domain.

## 7. Next Steps (Recommended)

1. Increase real receipt line data volume and diversity (more stores, fonts, lighting).
2. Add a clean held-out test set and report final metrics once.
3. If EasyOCR deployment is required, lock `custom.yaml` to the exact training config and validate loading via a small inference script.
