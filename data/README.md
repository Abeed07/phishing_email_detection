# Dataset

This directory holds the raw and processed email datasets.

## Expected format

`data/raw/emails.csv` must contain at minimum:

| column   | description                                  |
|----------|-----------------------------------------------|
| `text`   | the raw email body (subject + content)         |
| `label`  | `phishing` / `legitimate` (or `spam` / `ham`)  |

Column names are configurable in `config.yaml` under `columns:` if your
source dataset uses different names.

## Getting a real dataset

For genuine portfolio results, download one of these public sources and
save it as `data/raw/emails.csv` (renaming columns as needed):

- **Kaggle** — search "Phishing Email Detection" (several labeled
  datasets are available with permissive licenses)
- **Nazario Phishing Corpus** — a well-known archive of real phishing
  emails, commonly paired with a legitimate-email corpus
- **Enron Email Dataset** — widely used as the "legitimate" class
- **SpamAssassin public corpus** — pre-labeled ham/spam emails

## Demo / smoke-test dataset

If you just want to verify the pipeline runs end-to-end before wiring up
a real dataset, run:

```bash
python data/generate_sample_data.py
```

This generates a small synthetic dataset at `data/raw/emails.csv`. It is
template-based and intentionally easy to classify — **do not use results
from this dataset in your portfolio writeup**. Swap in a real dataset
before reporting final metrics.
