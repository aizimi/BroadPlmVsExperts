# Data

This directory contains dataset split files used in the paper experiments and small
metadata files. It does **not** contain raw dataset archives or private corpora
(except MACcorpus, which is distributed by the authors as open and free).

---

## Datasets Used

| Dataset | Language variety | Source |
|---|---|---|
| ASTD | Egyptian Arabic | Hugging Face: `arbml/ASTD` |
| ArSAS | Multi-dialectal Arabic | Hugging Face: `arbml/ArSAS` |
| AfriSenti_ARQ | Algerian Arabic | Hugging Face: `cardiffnlp/tweet_sentiment_multilingual` (arq) |
| AfriSenti_ARY | Moroccan Arabic | Hugging Face: `cardiffnlp/tweet_sentiment_multilingual` (ary) |
| MACcorpus | Moroccan Arabic | Included in `data/MACcorpus.csv` (see note below) |
| LABR | Multi-dialectal Arabic (reviews) | Hugging Face: `mohamedadaly/labr` |
| HARD | Hijazi Arabic (reviews) | Hugging Face: `Elnagara/hard` |

Datasets loaded from Hugging Face are downloaded automatically by the data loaders
on first run. No manual download is required for those.

---

## MACcorpus

`data/MACcorpus.csv` is included in this repository because the official MAC project
repository presents the corpus as open and free for research use.

**No explicit Creative Commons or other open-source license file was found** in the
original release. Users should treat this corpus accordingly and must cite the
original authors in any publication:

> Garouani, M., and Kharroubi, J. (2022). MAC: An Open and Free Moroccan Arabic
> Corpus for Sentiment Analysis. In *Innovations in Smart Cities Applications
> Volume 5*, Lecture Notes in Networks and Systems, vol. 393, Springer, Cham,
> pp. 849–858. DOI: 10.1007/978-3-030-94191-8_68.

---

## Reproducibility Splits

`data/splits/` contains JSON files that record the train / validation / test indices
used in the paper experiments. All splits were created with `split_seed=42`.

| File | Dataset |
|---|---|
| `afrisenti_arq_split_seed_42.json` | AfriSenti_ARQ |
| `afrisenti_ary_split_seed_42.json` | AfriSenti_ARY |
| `arsas_split_seed_42.json` | ArSAS |
| `astd_split_seed_42.json` | ASTD |
| `hard_split_seed_42.json` | HARD |
| `labr_split_seed_42.json` | LABR |
| `mac_split_seed_42.json` | MACcorpus |

These files are loaded automatically by the data loaders when a matching split file
exists. The loaders fall back to a fresh stratified split (with `split_seed=42`) if
the file is absent.

---

## Dataset Placement

| Dataset | Expected location | Notes |
|---|---|---|
| MACcorpus | `data/MACcorpus.csv` | Included in repo |
| ASTD | Downloaded from HF Hub | Pass `--csv` flag to use a local CSV |
| ArSAS | Downloaded from HF Hub | Pass `--csv` flag to use a local CSV |
| LABR | Downloaded from HF Hub | Pass `--csv` flag to use a local CSV |
| HARD | Downloaded from HF Hub | Pass `--csv` flag to use a local CSV |
| AfriSenti_ARQ / ARY | Downloaded from HF Hub | Automatic |

---

## License and Citation Requirements

Each dataset carries its own license and citation requirements. Users must consult
the original papers and Hugging Face dataset cards for each dataset and comply with
those terms. The split files and metadata in this directory are released alongside
the paper code to support reproducibility.
