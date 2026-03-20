# Data

## How to obtain the dataset

1. Go to: https://www.kaggle.com/datasets/rmisra/news-category-dataset
2. Sign in to Kaggle (free account required)
3. Click **Download** to get `archive.zip`
4. Extract and rename the file to `News_Category_Dataset_v3.json`
5. Place it in this `data/` folder

### Kaggle CLI (alternative)

```bash
pip install kaggle
kaggle datasets download -d rmisra/news-category-dataset
unzip news-category-dataset.zip -d data/
```

## File format

JSON-lines: one article per line.

```json
{
  "link": "https://www.huffpost.com/entry/...",
  "headline": "Article headline text",
  "category": "POLITICS",
  "short_description": "Brief description...",
  "authors": "Author Name",
  "date": "2018-05-26"
}
```

## Processed splits

After running `python src/data_pipeline.py` (or `bash run.sh`), the following files are created in `data/processed/`:

| File | Description |
|------|-------------|
| `train.csv` | Training split (80%) |
| `val.csv` | Validation split (10%) |
| `test.csv` | Test split (10%) |
| `label_map.json` | Integer → category name mapping |

**No raw PII is stored in this repository.**  
Author names are excluded from all model features.

## License

The dataset is released under **CC BY 4.0**.  
Citation: Misra, Rishabh. "News Category Dataset." Kaggle, 2022.
