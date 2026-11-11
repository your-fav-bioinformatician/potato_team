# Building Age from Space — Potato Team

**Sara S. Ali & Mostafa Mohammed** · Hack4Dev, November 2026

A building-age classifier that learns on Madrid from 42 years of Landsat time series (1984–2025) and adapts to any
city from a few labelled pixels. Each 30 m building pixel is assigned one of four construction periods, with a
confidence level.

## Results

| City | Labelled pixels per class | Method | Macro-F1 (mean ± sd) |
|---|---|---|---|
| Madrid | 10-fold spatial CV | backbone | 0.649 ± 0.038 |
| Amsterdam | n = 5 | bare backbone | 0.651 ± 0.017 |
| Amsterdam | n = 25 | full pipeline | 0.685 ± 0.028 |
| Amsterdam | n = 50 | full pipeline | 0.703 ± 0.022 |
| Amsterdam | n = 100 | full pipeline | 0.738 ± 0.007 |
| Amsterdam | n = 200 | full pipeline | 0.760 ± 0.006 |

Amsterdam: 10 random draws per n (seed 0), scored on every other labelled pixel. Average over n: 0.707.
Full method and results: [`docs/Potato_Team_Building_Age_Report.pdf`](docs/Potato_Team_Building_Age_Report.pdf) ·
slides: [`docs/Potato_Team_Building_Age_Presentation.pptx`](docs/Potato_Team_Building_Age_Presentation.pptx).

## Contents

| Path | What it is |
|---|---|
| `building_age_pipeline.py` | Inference script: unpacking, preprocessing, `modelCreate`, command line |
| `training/building-age-rebuild.ipynb` | Training notebook (Kaggle, GPU) that produced `model/`: from the raw parquet files to the backbone |
| `model/backbone.pt` | Stage-1 model: ensemble of 5 TempCNN embedding networks trained on Madrid, with normalisation statistics |
| `model/backbone_meta.json` | Backbone metadata: channel names, year axis, class boundaries, members used |
| `predictions/amsterdam_predictions.csv` | Predicted period and confidence for every Amsterdam building pixel |
| `predictions/madrid_predictions.csv` | Same for Madrid |
| `docs/` | Final report (PDF) and presentation (PPTX) |

## Running it

Requires Python with `numpy`, `pandas`, `pyarrow`, `torch`, `scikit-learn` (and `matplotlib` for the evaluation plot).
The script finds `model/backbone.pt` next to itself.

```bash
# a model for a city from n labelled pixels per class drawn from the dataset's own labels (fixed seed)
python building_age_pipeline.py --data amsterdam_data.parquet --bounds 1945,1984,2004 --centre 52.3731,4.8932 --n 25 --seed 0 --out out/

# the task's evaluation: 10 draws for each n in 5, 25, 50, 100, 200 -> mean ± sd, table, F1 vs log2(n) plot
python building_age_pipeline.py --data amsterdam_data.parquet --bounds 1945,1984,2004 --centre 52.3731,4.8932 --evaluate --out eval/

# your own labels (csv with px_key, py_key and age_class, or year + --bounds)
python building_age_pipeline.py --data city.parquet --labels labels.csv --out out/
```

`--data` takes a parquet file, a folder with one parquet, or a `.zip`. With n ≤ 20 labelled pixels per class the
backbone is used as it is (nearest class prototype); above that, the full pipeline runs (AdaBN, cross-validated
fine-tuning, confidence rule with spatial neighbours).

## Training

`training/building-age-rebuild.ipynb` is the Kaggle notebook that trained the backbone in `model/` (Kaggle T4 GPU,
about 1.5 h). Its only input is a Kaggle dataset holding the two raw files, `madrid_train.parquet` and
`amsterdam_data.parquet`; the code is written into the notebook itself. It runs:

1. Madrid preprocessing from the raw parquet (best observation per band, quality flags, gap filling, indices,
   3×3 neighbourhood spread) and the 10-fold spatial-block splits (~1 km blocks).
2. The backbone: 10 seeds × 10 folds of TempCNN with supervised contrastive loss (Madrid cross-validation), then
   10 networks trained on all of Madrid.
3. Pre-registered rules on that cross-validation: best checkpoint vs weight averaging, and the number of ensemble
   members (5) written to `backbone_meta.json`.
4. The transfer evaluation on Amsterdam.

## Prediction files

Columns:

- `id`: pixel id, `<City>_<px_key>_<py_key>` (the dataset's `pixel_id` without the year suffix)
- `label`: predicted construction period, 1–4: 1 = before the city's event year (1960 Madrid, 1945 Amsterdam),
  2 = event year–1984, 3 = 1984–2004, 4 = 2004–2024
- `confidence`: `high` (clear margin between the two closest periods), `moderate-high` (decided by labelled and
  high-confidence spatial neighbours), `low` (unclear and no labelled neighbour), or `labelled` (one of the labelled
  pixels given to `modelCreate`; its own label is reported)

Both files come from `modelCreate` with n = 200 labelled pixels per class (seed 0). The Madrid file is in-sample: the
backbone was trained on all of Madrid, so its Madrid predictions are not an independent test; the honest Madrid score
is the cross-validated one above.
