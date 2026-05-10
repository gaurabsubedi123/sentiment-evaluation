# Evaluating the Use of Sentiment Analysis for Autobiographical Text

Code and aggregate results for the ICSEng 2026 paper of the same name
(Takahashi, Subedi, Fonseca; UNLV).

We benchmark seven off-the-shelf sentiment / emotion classifiers
(TextBlob, VADER, nlptown mBERT, Cardiff RoBERTa, tabularisai mBERT,
Hartmann DistilRoBERTa, Bhadresh DistilBERT) plus an unweighted
ensemble and a TF-IDF + Logistic Regression baseline against
participant-rated valence on 2,730 autobiographical memories from 390
participants.

## Setup

Tested with Python 3.10. A GPU is recommended for the transformer
inference step but not required.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

## Run

Open `sentiment_analysis_full_pipeline.ipynb` and run all cells. The
notebook expects to be launched from the repo root (it sets
`BASE = os.path.abspath(".")`). End-to-end runtime is roughly 30-45
minutes on an L40 GPU; CPU is several hours.

The transformer checkpoints are pulled from the HuggingFace Hub on
first run. We did not pin a `revision`; results in the paper correspond
to the latest `main` branches as of 2026-05-01.

## Data

The autobiographical memory corpus (`data/forcaps_with_corrected.csv`)
is **not included**. It is the property of the Reasoning and Memory
(RAM) Lab at the University of Nevada, Las Vegas. Researchers interested
in obtaining the data should contact the lab directly.

Aggregate results from the paper (per-model metrics, statistical test
outputs, figures) are provided under `results/` so that the headline
numbers and figures are reproducible without the underlying data. The
per-memory and per-participant CSVs the notebook produces (e.g.
`memories_with_scores.csv`, `qualitative_examples.csv`,
`for_error_analysis*.csv`, `per_participant_accuracy_cardiff.csv`,
`conover_binary.csv`, `conover_continuous.csv`,
`conover_discretized.csv`) are intentionally omitted for the same
reason.

## Layout

```
.
├── sentiment_analysis_full_pipeline.ipynb   main notebook
├── requirements.txt
├── results/                                 aggregate CSVs from the paper
│   └── figures/                             PNGs used in the paper
├── LICENSE                                  MIT, code only
└── README.md
```


```
