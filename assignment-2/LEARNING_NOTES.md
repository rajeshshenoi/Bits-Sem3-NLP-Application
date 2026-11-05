# Learning notes – Information Extraction (PS3, CrossRE-AI)

Short explanations of every concept the notebook uses, in the order of the parts. Read the matching section
before running/discussing that part.

> Companion: [`STUDY_GUIDE.md`](STUDY_GUIDE.md) maps these concepts to the notebook cells, our results and a viva
> question bank.

---

## 1. Information extraction = NER + relation extraction

*Information extraction (IE)* turns free text into structured records.

```
"BERT, proposed by Jacob Devlin at Google, improved question answering."
   NER  →  [BERT]/algorithm  [Jacob Devlin]/researcher  [Google]/organisation  [question answering]/task
   RE   →  (BERT, Jacob Devlin) = origin?   (Jacob Devlin, Google) = role   (BERT, question answering) = usage?
```

For the consulting team these triples become a **technology map**: which algorithms are used for which tasks,
and who works on them.

## 2. Part A – span annotations → BIO tags

CrossRE stores entities as token spans `(start, end, type)` with an **inclusive** end.
Sequence labellers need one label per token, so we use **BIO**:

| token | Jacob | Devlin | at | Google |
| --- | --- | --- | --- | --- |
| tag | B-researcher | I-researcher | O | B-organisation |

* `B-X` begins an entity of type X, `I-X` continues it, `O` is outside.
* Off-by-one bug to avoid: span `(0, 1)` inclusive covers tokens 0 **and** 1 → exclusive end `2`.
* We check correctness with a *round trip*: spans → BIO → spans must be identical for every sentence.

**Why EDA matters:** class imbalance (some types have <60 examples), tiny data (450 sentences) and train/test shift
explain most of the later errors – cite them in the reflection.

## 3. Evaluation – entity-level P/R/F1 (seqeval)

An entity counts as correct only if **both boundaries and the type** match exactly.

* Precision = correct predicted entities / predicted entities
* Recall = correct predicted entities / gold entities
* F1 = harmonic mean. Token accuracy is misleading because ~70 % of tokens are `O`.
* `seqeval` default mode is CoNLL-style (a stray `I-X` starts an entity); `mode="strict", scheme=IOB2` treats invalid
  sequences strictly. Part G reports both.
* **micro** average (headline) weights every entity equally; **macro** averages types equally (rare types count more).

## 4. Part B – Conditional Random Field (CRF)

A linear-chain CRF models `P(tag sequence | sentence)` = exp(Σ feature weights) / Z.
Two kinds of weights:

* **state features** – "word ends in *-tion* and tag is B-task" (`crf.state_features_`)
* **transition features** – "I-task after B-task is good, I-task after O is bad" (`crf.transition_features_`)

Decoding uses the Viterbi algorithm to find the best whole sequence, so tags stay consistent.
*You* design features (that is the "classical" part): shape, case, affixes, POS, neighbouring words.
`c1` (L1) and `c2` (L2) are regularisation strengths – L1 drops useless features, L2 shrinks weights.
**Ablation** (remove one feature group, re-measure) is the honest way to show which features help.

## 5. Part C – BiLSTM-CRF with GloVe

```
word → GloVe vector (100-d) ┐
                            ├→ concat → BiLSTM (reads left→right and right→left) → linear → CRF → tags
chars → char-CNN (50-d)     ┘
```

* **GloVe**: word vectors pre-trained on 6 B tokens of Wikipedia+Gigaword – similar words have similar vectors,
  so the model knows *"perceptron"* ≈ *"classifier"* without seeing it labelled. Lower-cased → case lost.
* **char-CNN** learns from spelling, recovering case/morphology and handling out-of-vocabulary words.
* **BiLSTM** gives each token a representation that depends on the whole sentence.
* **CRF layer** = same transition modelling as Part B, but on learned features. Loss = negative log-likelihood.
* **Loss curves**: training loss keeps falling; when validation loss turns up, the model starts to over-fit → early
  stopping keeps the best epoch.

## 6. Part D – Fine-tuning a transformer (SciBERT)

* **BERT** is pre-trained (masked-language modelling) on huge text; **SciBERT** on 1.14 M scientific papers with its
  own vocabulary – much closer to AI-paper text. Fine-tuning = add a linear layer per token and train everything for a
  few epochs at a small learning rate (5e-5).
* **Sub-word tokenisation**: WordPiece splits rare words into pieces (`back ##propagation`). Labels are per word, so:
  1. special tokens (`[CLS]`, `[SEP]`) → `-100`,
  2. first piece of a word → the word's label,
  3. remaining pieces → `-100`.
  `-100` is PyTorch's `ignore_index`, so those positions add no loss. At prediction time we read the first piece of
  each word back → one tag per word → seqeval. `tokenizer(..., is_split_into_words=True).word_ids()` gives the
  piece → word map.
* **Why it usually wins**: contextual representations + domain pre-training carry knowledge the 450 sentences can't.

## 7. Part E – LLM-based NER (prompting)

* **Zero-shot**: instructions + type definitions only. **Few-shot**: also k worked examples (in-context learning).
  No weights change.
* Force **structured output** (JSON array of `{"text", "type"}`), then **ground** each string back to token positions
  – LLMs return text, not indices, so matching (whitespace/case-insensitive) is needed; unmatched or invalid output is
  an error, and unparseable responses must be counted.
* Typical LLM weaknesses on CrossRE: does not know the dataset's annotation guidelines (what is `misc`? `researcher`
  vs `person`?), paraphrases spans → boundary errors, slow and costly per sentence. Strength: no labelled data needed.
* Local model `Qwen2.5-3B-Instruct` (open weights) runs free on a Colab T4 and keeps client text private; a bigger
  API model would score higher but costs money per token and sends data outside.

## 8. Part F – Relation extraction (light)

* Given gold entities, classify every **ordered** pair (A, B) into a relation or `NO_RELATION`.
  Most pairs are unrelated → imbalance → we cap `NO_RELATION` at 2× positives and report **macro-F1**
  (every class counts equally, so the model cannot win by predicting `NO_RELATION`).
* **Method 1 – logistic regression** on hand-crafted features: entity types, words between, distance, order.
* **Method 2 – SciBERT with entity markers**: insert `[E1] … [/E1]` and `[E2] … [/E2]` around the arguments so the
  transformer knows *which* pair to classify; classify from the `[CLS]` representation.
* **Confusion matrix**: rows = gold, columns = predicted; off-diagonal cells show which relations get confused.

## 9. Part G – Fair comparison and error analysis

* Compare all models on the **same** 100 test sentences (LLMs were only run on those).
* Error categories: **boundary** (right type, wrong span), **type confusion** (right span, wrong type),
  **missed** (no overlap – especially rare types), **spurious** (predicted where gold has nothing).
* Reflection = accuracy × cost × data size × domain → a recommendation the numbers support.
