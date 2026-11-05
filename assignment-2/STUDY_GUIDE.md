# Study guide – Assignment 2, PS3 (Information Extraction on CrossRE‑AI)

A systematic path through `notebooks/Assignment2_Programming_Group_81.ipynb` (“the notebook”) and
`LEARNING_NOTES.md` (“the notes”). The notes explain **concepts**; this guide explains **the code, the design
decisions and our numbers**, and tests you with questions. Cell numbers (`[c17]`) are 0‑based positions in the
notebook (cell 0 = title); open the notebook on GitHub or in Colab next to this file.

---

## How to use this guide – 6 study sessions

| Session | Read | Do | You are done when you can… |
| --- | --- | --- | --- |
| 1 · Big picture (45 min) | §1–§3 of this guide, notes §1–§3 | Skim the whole notebook once, only headings and outputs | explain the business problem, the 7 parts and why entity‑level F1 is the metric |
| 2 · Data (60 min) | §4 Part A, notes §2 | Trace one sentence through `prepare()` by hand | convert an inclusive span to BIO on paper and explain the dev split |
| 3 · Classical + neural (90 min) | §4 Parts B–C, notes §4–§5 | Exercises E1–E3 | justify every CRF feature group and draw the BiLSTM‑CRF |
| 4 · Transformers + LLMs (90 min) | §4 Parts D–E, notes §6–§7 | Exercises E4–E6 | explain `-100`, `word_ids()`, few‑shot vs zero‑shot results |
| 5 · Relations + comparison (60 min) | §4 Parts F–G, notes §8–§9 | Exercise E7 | read both confusion matrices and the error categories |
| 6 · Viva practice (60 min) | §6 question bank | Answer aloud **before** opening each answer | answer 25/30 without looking |

---

## 1. What the assignment is really testing

**Business story (PS3).** An IT‑services company’s technology‑scouting team wants to read AI papers automatically:
find **entities** (algorithms, tasks, fields, products, researchers, …) and **relations** between them (*part‑of*,
*usage*, …) to build a technology map.

**Pedagogical intention.** The assignment is a *ladder of increasing complexity* on one fixed dataset, so you can
see what each generation of NLP buys you:

```
hand-made features      →  learned features          →  pre-trained context        →  no training at all
CRF (Part B)               BiLSTM-CRF + GloVe (C)        SciBERT fine-tuning (D)        LLM prompting (E)
```

plus a second IE step (relation extraction, F) and a **fair, evidence‑based comparison** (G). The grader checks
that you (1) implemented each rung correctly, (2) measured all of them the **same way**, and (3) can argue which one
an organisation should use, with *cost, data size and domain* in mind.

**The integrity rules and why they exist** (brief §4) – know these, they explain much of the code:

| Rule | Why it exists | Where it shows up in the code |
| --- | --- | --- |
| Entity‑level P/R/F1 with `seqeval`, not token accuracy | ~70 % of tokens are `O`; a model predicting only `O` would get high token accuracy | every `classification_report(...)` |
| B, C, D on the **full test set**; E on a **fixed sample ≥100** | LLMs are slow/expensive, so they get a sample – but then everyone must be re‑scored on it | `sample_idx` [c32], re‑slicing in [c45] |
| Part G numbers must come from variables | stops hand‑typed (fabricated) tables | `comparison` DataFrame [c45], auto summary [c49] |
| No mock LLM | a rule‑based fake “LLM” scores zero | real `Qwen2.5-3B-Instruct` [c35] |
| `NO_RELATION` capped at 2× positives | otherwise ~88 % of pairs are negatives and the classifier learns “always NO_RELATION” | `build_pairs()` [c40] |
| Executed notebook, no errors | proves the numbers are real | Colab run committed to the repo |

**Rubric → where the marks are earned**

| Part (marks) | “Full marks” needs | Our evidence |
| --- | --- | --- |
| A (1.0) | correct BIO, verified; complete EDA | round‑trip check = 0 mismatches; 3 printed sentences; counts, types, lengths [c9–c11] |
| B (1.5) | CRF + justified features + per‑type report | grid + **ablation** table [c16], report + learned weights [c18] |
| C (1.5) | BiLSTM(-CRF) + pretrained embeddings + loss curve + report | GloVe 400k vectors [c21], curve + report [c24] |
| D (2.0) | fine‑tuned; alignment **correct and explained** | alignment table [c27], explanation [c30] |
| E (1.5) | zero+few‑shot, JSON→spans, CSV, failures counted | [c32–c37], `outputs/llm_ner_outputs.csv` |
| F (1.5) | baseline + 2nd method, macro‑F1, confusion matrix | [c40–c42] |
| G (1.0) | code table on shared sample, 10 errors, reflection matching numbers | [c45–c50] |

---

## 2. The data contract – what flows through the notebook

Understanding these few objects makes every later cell easy to read.

```python
# raw record (after loading, [c7])
{"doc_key": "ai-train-1",
 "tokens":  ["Popular", "approaches", "of", "opinion-based", "recommender", "system", ...],
 "ner":     [(3, 5, "product"), (10, 11, "field"), ...],          # (start, end_INCLUSIVE, type)
 "relations": [(3, 5, 10, 11, "part-of"), (3, 5, 10, 11, "usage"), ...]}   # (e1 span, e2 span, label)

# prepared record (after prepare(), [c9])
{"doc_key": ..., "tokens": [...],
 "tags":  ["O", "O", "O", "B-product", "I-product", "I-product", ...],     # one BIO tag per token
 "spans": [(3, 5, "product"), ...],                                         # still inclusive
 "relations": {(3, 5, 10, 11): "part-of", ...},                             # FIRST label only
 "n_dup_rel": 1,
 "pos": ["JJ", "NNS", "IN", ...]}                                           # added in Part B [c14]
```

**Key globals you will meet everywhere**

| Name | Meaning | Size |
| --- | --- | --- |
| `train_data` | train + validation, prepared | 450 sentences |
| `test_data`, `test_tokens`, `test_tags` / `y_test` | test split | 431 sentences, 1,811 entities |
| `fit_ids`, `dev_ids` (`fit_set`, `dev_set`) | 90/10 split of `train_data` for model selection only | 405 / 45 |
| `ENTITY_TYPES` | types seen in train+val | 15 (`event` exists only in test) |
| `tag_list`, `tag2id` / `label2id` | BIO label inventory | 31 = `O` + 15 types × {B, I} |
| `sample_idx` | the fixed 100 test sentences used in E and G | seed 42 |
| `y_pred_crf`, `y_pred_lstm`, `y_pred_bert` | word‑level predictions on the full test set | lists of tag lists |
| `llm0_sample`, `llmk_sample` | LLM predictions on the sample | 100 tag lists each |
| `TIMINGS` | train seconds + inference ms/sentence per model | used in Part G |

**Notebook map**

| Cells | Part | Main functions / objects |
| --- | --- | --- |
| c0–c2 | Title, group, problem & approach | – |
| c4–c5 | Setup | `SEED=42`, `DEVICE`, `OUT_DIR`, `TIMINGS` |
| c7–c12 | A Data/EDA | `_from_hf`, `_from_github`, `spans_to_bio`, `bio_to_spans`, `prepare`, `show_tagged` |
| c14–c19 | B CRF | `pos_tags`, `shape`, `word2features`, `train_crf`, ablation, `crf.state_features_` |
| c21–c25 | C BiLSTM‑CRF | `get_glove`, `norm_word`, `BiLSTMCRF`, `make_batch`, `train_bilstm` |
| c27–c30 | D SciBERT | `tokenize_and_align`, `to_hf`, `word_level_predict`, `make_args`, `DevF1Callback`, `fine_tune` |
| c32–c38 | E LLM | `parse_llm_json`, `llm_json_to_bio`, `SYSTEM_PROMPT`, `pick_few_shot`, `build_messages`, `run_llm` |
| c40–c43 | F Relations | `build_pairs`, `pair_features`, `LogisticRegression`, `marked_text`, SciBERT classifier |
| c45–c50 | G Comparison | `comparison`, `per_type`, `categorise`, error picker, auto summary |

---

## 3. Results cheat sheet (memorise these)

| Model | Test P | Test R | **Test F1** | Sample F1 (100 sent.) | Train time | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CRF | 0.613 | 0.551 | **0.581** | 0.562 | 24 s (CPU) | 2.9 ms/sent |
| BiLSTM‑CRF + GloVe | 0.632 | 0.628 | **0.630** | 0.581 | 122 s | 2.7 ms |
| SciBERT (cased) | 0.703 | 0.765 | **0.732** | 0.722 (strict 0.744) | 39 s (T4) | 1.2 ms |
| LLM zero‑shot (Qwen2.5‑3B) | – | – | – | 0.378 (P 0.607, R 0.275) | 0 | 865 ms |
| LLM few‑shot (5 ex.) | – | – | – | 0.547 (P 0.567, R 0.528) | 0 | 1,614 ms |

Relation extraction (macro‑F1, 7 classes): **LogReg 0.523**, **SciBERT + markers 0.530**.
SciBERT errors on the sample: 128 = 39 type confusion, 31 boundary, 23 boundary+type, 20 spurious, 15 missed.
Hardest NER type for everyone: `misc` (≤ 0.33). Hardest relation: `related-to` (≈ 0.25).

---

## 4. Part‑by‑part deep dive

Each part follows the same pattern: **goal → walk through the code → concept → our result → pitfalls → self‑check.**

### Part A – Data preparation and EDA  `[c7–c12]` · notes §2–§3

**Goal.** Turn span annotations into per‑token BIO tags, prove the conversion is right, describe the data.

**Code walk‑through**
1. **Loading `[c7]`.** `_from_hf()` tries `load_dataset("DFKI-SLT/cross_re", "ai", revision="refs/convert/parquet")`.
   HF can return a list column either as a *list of dicts* or a *dict of lists*; `_rows()` normalises both, and
   `_label_names()` converts integer `ClassLabel`s to strings. If the Hub fails, `_from_github()` reads the identical
   JSON files from the authors’ repo (in our Colab runs the fallback was used – the printed source line says so).
2. **BIO `[c8]`.** `spans_to_bio(n, spans)` expects **exclusive** ends; `bio_to_spans()` is its inverse.
3. **`prepare()` `[c9]`** – the most important 15 lines of Part A:
   * `spans_excl = [(s, e + 1, t) ...]` – **the inclusive → exclusive fix**. Forgetting `+1` silently drops the last
     token of every entity (single‑token entities would vanish entirely).
   * Relations: a dict keyed by `(s1, e1, s2, e2)`; the first label wins, later ones are counted in `n_dup_rel`.
   * Verification: for every sentence, `bio_to_spans(tags) == gold spans` → **0 mismatches** in all splits.
4. **Splits.** `train_data = train + validation` (brief rule). `dev_ids` = 10 % of it (seed 42) for model selection.
5. **EDA `[c10–c11]`.** Counts per split, type distribution (+ bar chart), average entity length, 3 tagged sentences.

**Concept check.** Why a dev split if the brief says “train on train+val”? Because hyper‑parameters (CRF `c1/c2`,
number of epochs) must be chosen on data that is **not** the test set – otherwise the test score is optimistic.
We select on dev and then **refit on all 450 sentences** (C and D), so no training data is wasted.

**Our numbers.** 450 train sentences, 14,701 tokens, 2,080 entities; test 431 / 12,991 / 1,811. 32.7 tokens and
4.6 entities per sentence. 15 types; `academicjournal` has 2 examples, `event` appears only in test. 86 duplicate
relation labels dropped in train+val.

**Pitfalls.** inclusive ends; assuming one relation label per pair; tuning on test.

**Self‑check.** Span `(16, 17, "task")` in a 20‑token sentence → which indices get which tags?
<details><summary>Answer</summary>Exclusive span (16, 18): tag[16] = B‑task, tag[17] = I‑task, all others unaffected.</details>

### Part B – CRF  `[c14–c19]` · notes §4

**Goal.** A strong classical baseline with hand‑crafted, *justified* features.

**Code walk‑through**
* `[c14]` POS tags with NLTK’s perceptron tagger, computed once and cached in `d["pos"]`.
* `[c15]` Features are organised in **six groups** – `lexical, case, shape, affix, pos, context` – so they can be
  switched off one at a time. `short_shape("ResNet-50") → "XxXx-d"`‑style patterns generalise to unseen words.
  `context` repeats the token features for positions −2…+2 and adds word bigrams.
* `[c16]` `train_crf()` = `sklearn_crfsuite.CRF(algorithm="lbfgs", c1, c2, all_possible_transitions=True)`.
  - **Grid** over 5 `(c1, c2)` pairs on dev → best `c1 = c2 = 0.1` (dev F1 0.594).
  - **Ablation**: remove one group → dev F1 drop: context −0.050, affix −0.037, case −0.023, lexical −0.018,
    POS −0.015, shape −0.011. *This table is your feature justification.*
* `[c17–c18]` Final CRF on all 450 sentences; per‑type report; strongest **state features** (e.g.
  `B-metrics ← -2:word=rate`) and **transitions** (`O → I-task` strongly negative = learned BIO grammar).

**Concept.** A linear‑chain CRF scores a *whole tag sequence*: Σ state‑feature weights + Σ transition weights, and
Viterbi decoding finds the best sequence. `c1` = L1 (sparsity), `c2` = L2 (shrinkage).

**Our result.** F1 0.581 (P 0.613 > R 0.551 – conservative). Good: `field` 0.787, `metrics` 0.716. Bad: `person`
0.154, `misc` 0.234, `organisation` 0.483.

**Why recall is low.** Sparse lexical features only fire for words seen in training; an unseen algorithm name
has only shape/affix/context evidence.

### Part C – BiLSTM‑CRF with GloVe  `[c21–c25]` · notes §5

**Goal.** Learn features automatically, starting from pre‑trained word vectors; show the loss curve.

**Code walk‑through**
* `[c21]` `get_glove()` downloads GloVe 6B‑100d (HF mirror → Stanford → warning). We got 400k vectors.
* `[c22]` Vocabulary = training words ∪ GloVe words occurring in the corpus (unsupervised; no test labels used);
  digits normalised to `0`. Embedding matrix initialised from GloVe; 97.5 % of test tokens covered.
* `[c23]` `BiLSTMCRF`:
  ```
  words → Embedding(GloVe,100d, fine-tuned) ┐
  chars → Embedding(30d) → Conv1d(50, k=3) → max-pool ┘→ concat(150) → dropout → BiLSTM(2×100) → dropout → Linear(31) → CRF
  ```
  `loss()` = negative log‑likelihood from `torchcrf.CRF`; `decode()` = Viterbi. `make_batch()` pads words, chars
  (max 20 per word) and builds a boolean `mask` so padding is ignored by the CRF.
* `train_bilstm()` – Adam 1e‑3, batch 16, grad‑clip 5; with `dev` it records train/val loss and dev F1, keeps the
  best state (early stopping, patience 10). **Phase 1** (fit/dev) chooses `BEST_LSTM_EPOCH = 42`; **phase 2**
  refits on all 450 sentences for 42 epochs.

**Reading the loss curve.** Train and val loss fall together until ~epoch 12; val loss flattens ~12 from epoch
~25 while train loss → 2: over‑fitting gap. Dev F1 still creeps up to 0.636 at epoch 42 → we select by F1, the
metric we report, not by loss.

**Our result.** F1 0.630 (+0.049 vs CRF), gain mostly **recall** (0.551 → 0.628). Big wins on name‑like types
(`person` 0.154 → 0.538, `country` 0.571 → 0.857). Only `metrics` got worse (0.716 → 0.687).

**Why the char‑CNN?** GloVe 6B is lower‑cased, so capitalisation (a strong NER cue) would be lost; characters
restore case and morphology and handle out‑of‑vocabulary technical words.

### Part D – SciBERT fine‑tuning  `[c27–c30]` · notes §6

**Goal.** Fine‑tune a domain‑pre‑trained transformer; implement and **explain** sub‑word alignment.

**Code walk‑through**
* `[c27]` `AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)` – **important fix**: the hub copy of
  `scibert_scivocab_cased` has no tokenizer config, so it would lower‑case input for a *cased* model. The printed
  `case preserved: ['Ten', '##sor', '##Flow', …]` proves the fix. (Before the fix test F1 was 0.678; after 0.732.)
* `tokenize_and_align()` – uses `enc.word_ids()`: `None` (special tokens) → `-100`; first sub‑token of a word →
  its label id; following sub‑tokens → `-100`. The alignment table shows e.g. `Multim` = B‑task, `##odal` = −100.
* `[c28]` `to_hf()` builds a HF `Dataset`; `DataCollatorForTokenClassification` pads `input_ids` *and* `labels`
  (padding labels = −100). `word_level_predict()` inverts the mapping at inference → one tag per original token,
  so seqeval compares exactly the same units as for the CRF/BiLSTM.
* `make_args()` builds `TrainingArguments` compatible with transformers v4 and v5 (renamed arguments).
  `DevF1Callback` computes word‑level dev F1 after every epoch.
* `fine_tune()` – lr 5e‑5, linear schedule, 10 % warm‑up, weight decay 0.01, batch 16, fp16, 20 epochs in
  phase 1 → best epoch 16 (dev F1 0.665) → `[c29]` phase 2 refit on all data for 16 epochs (39 s).

**Reading the curve.** Dev loss bottoms out at epoch ~4–7 and drifts up, while dev F1 keeps rising – the model
becomes over‑confident on the tokens it gets wrong (raising cross‑entropy) while getting more full entities right.

**Our result.** F1 **0.732** (P 0.703, R 0.765) – best model: +0.103 vs BiLSTM, +0.152 vs CRF; macro‑F1 0.662.
Biggest gains: `researcher` 0.912, `person` 0.785, `metrics` 0.854, `algorithm` 0.778. Still weak: `misc` 0.324.

**Self‑check.** Why not average the predictions of all sub‑tokens of a word?
<details><summary>Answer</summary>You could (an alternative scheme), but first‑sub‑token is the standard BERT recipe: one loss term per word,
no over‑weighting of long words, and the first piece carries the word start (important for B‑ vs I‑). The key is to
use the <i>same</i> rule in training and prediction.</details>

### Part E – LLM‑based NER  `[c32–c38]` · notes §7

**Goal.** Zero‑ and few‑shot prompting of a real LLM; JSON output mapped to spans; failures counted; outputs saved.

**Code walk‑through**
* `[c32]` `sample_idx = sorted(random.Random(SEED).sample(range(431), 100))` – the shared sample (422 entities).
* `[c33]` `parse_llm_json()` strips ```` ```json ```` fences, parses the outermost `[...]`; if that fails, salvages
  complete `{...}` objects (status `salvaged`), else `failed`. `llm_json_to_bio()` maps each entity **text** back to
  token positions: whitespace‑insensitive concatenation matching, longest entities first, first free occurrence;
  type synonyms normalised (`organization → organisation`); unknown types dropped. A unit test shows it works.
* `[c34]` `SYSTEM_PROMPT` = role + 15 types with one‑line definitions + “return ONLY a JSON array”.
  `pick_few_shot()` greedily picks 5 short training sentences covering the most types (all 15 covered);
  `build_messages()` adds them as user/assistant turns with gold JSON.
* `[c35]` `Qwen/Qwen2.5-3B-Instruct` in fp16; **left padding** (needed for batched generation with decoder‑only
  models), greedy decoding (`do_sample=False`, reproducible), batch 8, max 512 new tokens. Every prompt, raw
  response, parse status, token counts and latency go to `outputs/llm_ner_outputs.csv`.
* `[c36]` seqeval on the sample per setting + `llm_summary`; `[c37]` `llm_behaviour` counts empty answers, types used.

**Our result.**
| | P | R | F1 | empty `[]` answers | parse failures | latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero‑shot | 0.607 | 0.275 | 0.378 | 49/100 | 0 | 0.87 s |
| few‑shot | 0.567 | 0.528 | 0.547 | 8/100 | 0 | 1.61 s |

Interpretation: format compliance is perfect; the problem is **annotation granularity**. Zero‑shot doesn’t know
that “information retrieval” counts as an entity in CrossRE and never uses `researcher` (calls them `person`).
Few‑shot examples teach the guideline implicitly (+0.169 F1) but cost 2.7× longer prompts. `misc` = 0 in both.

### Part F – Relation extraction  `[c40–c43]` · notes §8

**Goal.** Classify ordered pairs of gold entities; two methods; macro‑F1 + confusion matrices.

**Code walk‑through**
* `[c40]` `TOP5 = part-of, named, physical, related-to, role` (from train+val); others → `OTHER`.
  `build_pairs()` creates **all ordered pairs** (A, B) ≠ (B, A); unannotated → `NO_RELATION`, randomly
  down‑sampled to 2× the positives (train 1,356 pos → 2,712 neg kept of 9,664; test 1,127 → 2,254 of 8,009).
* `[c41]` `pair_features()`: both types + type pair, argument order, distance bucket, #entities between,
  bag of words between (+ first/last), entity words and heads, ±2 outside words → `DictVectorizer` (9,505 sparse
  features) → `LogisticRegression(class_weight="balanced")`. **macro‑F1 0.523**.
* `[c42]` SciBERT sequence classifier on text with **typed entity markers**
  (`… [E1] algorithm | recurrent neural network [/E1] … [E2] misc | … [/E2]`); markers added as special tokens
  (`resize_token_embeddings`), 6 epochs, lr 3e‑5, `[CLS]` classification head. **macro‑F1 0.530**.

**Reading the confusion matrices.** Rows = gold, columns = predicted. Biggest off‑diagonal mass in both: real
relations → `NO_RELATION`, and `OTHER` ↔ `part-of`. SciBERT rejects more false links (1,949 vs 1,866 correct
`NO_RELATION`) but also misses more real ones (e.g. `physical → NO_RELATION` 38 vs 17) because it was trained
without class weights.

**Why macro‑F1?** `NO_RELATION` is 2/3 of the data; micro/accuracy would reward predicting it. Macro averages the 7
classes equally.

### Part G – Comparison, errors, reflection  `[c45–c50]` · notes §9

* `[c45]` Re‑slices CRF/BiLSTM/SciBERT predictions to `sample_idx`, adds both LLM settings → one `comparison` table
  with P/R/F1, strict‑IOB2 F1, full‑test F1, train time, inference ms. **Built from variables – never typed.**
* `[c46]` Per‑type F1 on the sample for every model (+ support and training counts).
* `[c47]` `categorise()` compares gold vs predicted spans per sentence:
  exact span + wrong type → *type confusion*; overlap + same type → *boundary*; overlap + different type →
  *boundary + type*; no overlap → *missed* (or *missed rare type* if <100 training examples); prediction with no
  gold overlap → *spurious*.
* `[c48]` Picks 10 errors round‑robin across categories (deterministic) and prints tokens / gold / pred.
* `[c49]` Prints a Markdown summary from the variables – the reflection quotes it, so text and numbers agree.

**Two subtle points examiners like**
1. *Strict F1 > default F1 for SciBERT (0.744 vs 0.722).* SciBERT sometimes emits an entity starting with `I‑`
   (e.g. “digital assistants”). Default (CoNLL) mode turns that into an extra predicted entity → false positive;
   strict IOB2 mode discards ill‑formed chunks. A CRF layer on top of BERT would prevent invalid sequences.
2. *BiLSTM vs CRF looks closer on the sample (0.581 vs 0.562) than on the full test (0.630 vs 0.581).* 100 sentences
   is a small, noisy sample; the full test set is the more reliable estimate. Never over‑interpret small differences.

**The recommendation (and its evidence).** SciBERT for NER (best accuracy, cheap inference, trains in < 1 min on a
free GPU); a cheap relation classifier (LogReg ≈ SciBERT); the LLM as an annotation assistant only (needs no labels,
but 0.547 F1 and ~700–1,300× slower than SciBERT).

---

## 5. Cross‑cutting design decisions (be ready to defend each)

| Decision | Reason |
| --- | --- |
| One `SEED = 42` for every random step | reproducibility (sampling, splits, init); stated in the notebook |
| Select on dev, refit on all train+val | honest model selection + use all 450 sentences for the final model |
| Select epochs by **dev F1**, not dev loss | F1 is the reported metric; loss and F1 diverge (curves in C and D) |
| `seqeval` default + strict reported | default = CoNLL convention (template uses it); strict shows the effect of invalid BIO |
| Word‑level evaluation for SciBERT | same units as CRF/BiLSTM → fair comparison |
| Greedy decoding for the LLM | deterministic, reproducible outputs |
| Failed LLM parses scored as all‑`O` | penalised, not dropped – otherwise you could “improve” F1 by failing |
| First relation label kept | brief rule; turns a multi‑label problem into single‑label classification |
| Ordered pairs for RE | relations are directional (*A part‑of B* ≠ *B part‑of A*) |
| Timings recorded per model | the reflection must discuss cost with numbers |

---

## 6. Viva question bank (answer aloud first, then open)

<details><summary>1. What problem does the notebook solve, in one sentence?</summary>Extracting AI‑domain entities and their relations from research text (CrossRE‑AI) and comparing CRF, BiLSTM‑CRF, SciBERT and LLM prompting to recommend a system for a technology‑scouting team.</details>

<details><summary>2. Why is token accuracy not acceptable?</summary>Most tokens are <code>O</code>; a model that predicts only <code>O</code> scores high token accuracy but finds no entities. Entity‑level F1 requires exact boundaries and type.</details>

<details><summary>3. What does “inclusive end” mean and how did you handle it?</summary>CrossRE’s end index is part of the entity. We add 1 before <code>spans_to_bio</code> (exclusive convention) and verify via a round trip for every sentence (0 mismatches).</details>

<details><summary>4. How did you verify the BIO conversion?</summary>Round trip spans → BIO → spans equals gold for all 881 sentences, plus three printed sentences with each span’s tokens and tags.</details>

<details><summary>5. Why a dev split, and what did you do after selection?</summary>To tune hyper‑parameters/epochs without touching test. After selection the BiLSTM and SciBERT were refit on all 450 sentences.</details>

<details><summary>6. Which CRF features mattered most and how do you know?</summary>Context (±2 words) – removing it cost 0.050 dev F1 in the ablation; then affixes (−0.037) and case (−0.023).</details>

<details><summary>7. What are c1 and c2?</summary>L1 and L2 regularisation strengths of the CRF; selected 0.1/0.1 by a grid on dev.</details>

<details><summary>8. What do CRF transition weights represent? Give an example from our model.</summary>Scores for tag bigrams; <code>O → I‑task</code> ≈ −3.0, i.e. the model learnt that an I‑ tag cannot follow O.</details>

<details><summary>9. Why does the BiLSTM add a CRF layer instead of softmax per token?</summary>Softmax predicts each tag independently and can output invalid sequences; the CRF scores whole sequences with transitions and decodes with Viterbi.</details>

<details><summary>10. Why add a character CNN when you already use GloVe?</summary>GloVe 6B is lower‑cased and has OOV words; characters restore capitalisation and morphology (e.g. <i>TensorFlow</i>, <i>‑ing</i>).</details>

<details><summary>11. What does the BiLSTM loss curve show?</summary>Over‑fitting after ~25 epochs (val loss flat ~12, train loss → 2) while dev F1 still improves to 0.636 at epoch 42 → selection by F1.</details>

<details><summary>12. Where did the BiLSTM beat the CRF and why?</summary>Mostly recall (0.551 → 0.628): pre‑trained vectors give unseen words meaningful representations; big gains on person, country, university.</details>

<details><summary>13. Explain sub‑word alignment and −100.</summary>Labels are per word, WordPiece splits words; via <code>word_ids()</code> the first sub‑token gets the label, other sub‑tokens and special tokens get −100 (PyTorch <code>ignore_index</code>), so they add no loss. Prediction reads the first sub‑token of each word.</details>

<details><summary>14. Why SciBERT and why cased?</summary>Pre‑trained on scientific papers with a scientific vocabulary – close to AI text; cased because capitalisation signals names/products. We had to set <code>do_lower_case=False</code>; the fix raised test F1 from 0.678 to 0.732.</details>

<details><summary>15. Dev loss rises after epoch ~5 but you trained 16 epochs. Contradiction?</summary>No: loss measures confidence, F1 measures exact entities; dev F1 kept rising to 0.665 at epoch 16, and F1 is what we report.</details>

<details><summary>16. Why is SciBERT best?</summary>Contextual, domain‑pre‑trained representations: it resolves context‑dependent types (researcher vs person) and technical vocabulary that 450 sentences cannot teach – F1 0.732.</details>

<details><summary>17. How do you turn LLM output into a seqeval score?</summary>Force JSON <code>[{text,type}]</code>, parse it, normalise types, find each text in the tokens (whitespace‑insensitive, longest first), write BIO tags, score with seqeval.</details>

<details><summary>18. Why is zero‑shot recall so low?</summary>The model answered <code>[]</code> for 49/100 sentences: it doesn’t know CrossRE’s granularity (generic task/method phrases are entities) and never used <code>researcher</code>.</details>

<details><summary>19. What did few‑shot change, at what cost?</summary>F1 0.378 → 0.547, empty answers 49 → 8, but prompts 2.7× longer and latency 0.87 → 1.61 s/sentence; some invented off‑schema types.</details>

<details><summary>20. How did you count unparseable outputs, and how are they scored?</summary>Status ok / salvaged / failed from <code>parse_llm_json</code>; failed → all‑O predictions (penalised). We had 0 failures in both settings.</details>

<details><summary>21. Why left padding and greedy decoding?</summary>Decoder‑only models generate after the last token, so pads must be on the left for batching; greedy makes outputs deterministic.</details>

<details><summary>22. How are RE candidates built and why cap NO_RELATION?</summary>All ordered gold‑entity pairs per sentence; unannotated = NO_RELATION, capped at 2× positives (seed 42), otherwise ~88 % negatives would dominate.</details>

<details><summary>23. Why macro‑F1 for RE?</summary>Each of the 7 classes counts equally, so a model cannot score well by predicting the majority NO_RELATION class.</details>

<details><summary>24. What are entity markers?</summary>Special tokens <code>[E1]…[/E1]</code>, <code>[E2]…[/E2]</code> (with the type) around the two arguments so the transformer knows which pair to classify.</details>

<details><summary>25. Why did SciBERT barely beat LogReg in RE?</summary>Only ~1.4k positive pairs, CLS pooling and no class weights: it sends more real relations to NO_RELATION; LogReg was class‑balanced with strong lexical features. Fixes: weighted loss, marker pooling, epoch tuning.</details>

<details><summary>26. Which relation is hardest and why?</summary><code>related-to</code> (≈0.25): a vague catch‑all with no typical wording; <code>OTHER</code> merges 12 types and overlaps with part‑of.</details>

<details><summary>27. Why is SciBERT’s strict F1 higher than default?</summary>It sometimes predicts entities starting with I‑; default mode counts them as (wrong) entities, strict IOB2 discards them.</details>

<details><summary>28. What is the most common SciBERT error and an example?</summary>Type confusion (39 of 128), e.g. <i>Audio Engineering Society</i> gold conference → predicted organisation.</details>

<details><summary>29. Why must B, C, D be re‑scored on the 100‑sentence sample?</summary>The LLM only ran on the sample; comparing its sample score with others’ full‑test scores would not be like‑for‑like.</details>

<details><summary>30. What would you recommend to the organisation and why?</summary>SciBERT NER (best F1, 1.2 ms/sentence, trains in 39 s on a free GPU) + a cheap relation classifier; LLM only for pre‑annotation since it needs no labels but scores 0.547 and is ~700–1,300× slower; collect more labels for misc/rare types.</details>

<details><summary>31. Name three ways to improve the system.</summary>CRF layer on SciBERT (valid BIO), more labelled data for misc/rare types (LLM pre‑annotation + human review), class‑weighted / marker‑pooled relation model; also a larger LLM or ensembling.</details>

<details><summary>32. What is the risk of tuning on the test set?</summary>Over‑optimistic scores that don’t generalise – the reported number would no longer be an unbiased estimate.</details>

---

## 7. Hands‑on exercises (to go from “understand” to “expert”)

Make a copy of the notebook in Colab (File → Save a copy in Drive), change one thing, **predict the effect
first**, then run only the needed cells.

| # | Change | Predict, then check |
| --- | --- | --- |
| E1 | CRF: remove `context` *and* `affix` together | is the drop additive (≈ −0.09) or smaller? (features overlap) |
| E2 | CRF: add a gazetteer feature (word in a list of 20 known AI terms from train) | which types gain? risk of over‑fitting? |
| E3 | BiLSTM: `freeze=True` on the embedding, or remove the char‑CNN | which types lose most without the char‑CNN? (case‑dependent ones) |
| E4 | SciBERT: label **all** sub‑tokens (`I-X` for continuation pieces) | F1 similar? what must change in `word_level_predict`? |
| E5 | SciBERT: `bert-base-cased` instead of SciBERT | expect lower F1 – domain pre‑training effect |
| E6 | LLM: add 2 more few‑shot examples rich in `misc`, or add “researcher = person who does AI research” rule | does `misc`/`researcher` improve? latency? |
| E7 | RE: give SciBERT class weights (custom `Trainer.compute_loss`) | recall on positives ↑, NO_RELATION precision ↓, macro‑F1 ↑? |
| E8 | Change `SEED` to 7 and rerun D | how much does F1 move? (gives a feel for run‑to‑run variance) |

---

## 8. One‑page summary to revise the night before

* **Task:** NER (15 AI types, BIO) + light RE (top‑5 + OTHER + NO_RELATION) on CrossRE‑AI; 450 train / 431 test.
* **Metric:** seqeval entity‑level micro F1 (exact span + type); RE: macro‑F1 over 7 classes.
* **Ladder:** CRF 0.581 → BiLSTM‑CRF 0.630 → SciBERT **0.732**; LLM 0.378 zero‑shot / 0.547 few‑shot (sample).
* **Why each step helps:** features → learned dense vectors (recall) → context + domain pre‑training (everything);
  LLM: no labels needed but doesn’t know the guidelines.
* **Key techniques:** inclusive→exclusive spans, dev‑select‑then‑refit, ablation, char‑CNN, CRF transitions, −100
  alignment, JSON grounding, ordered pairs + negative capping, entity markers, shared sample, error taxonomy.
* **Recommendation:** SciBERT NER + cheap RE + LLM as annotation assistant; more labels for `misc`/rare types.

*Related files:* `LEARNING_NOTES.md` (concepts), `README.md` (plan & results), `outputs/llm_ner_outputs.csv`
(all prompts and raw LLM responses).
