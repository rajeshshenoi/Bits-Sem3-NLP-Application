# Assignment 2 – Information Extraction and Situated Learning

**Problem Statement 3: AI Research Intelligence – Methods, Tasks and Researchers**
Domain: technology research & consulting (technology-scouting team of an IT services company).
Total: 15 marks = Part 1 programming (10) + Part 2 situated learning (5).

## Folder layout

```
assignment-2/
├── README.md                     ← this plan + checklist
├── LEARNING_NOTES.md             ← the concepts behind every part, explained
├── brief/                        ← documents received from BITS (do not edit)
│   ├── Assignment2_IE_PS3.docx
│   └── Information_Extraction_Assignment_Template.ipynb
├── notebooks/
│   └── Assignment2_Programming_Group_81.ipynb   ← Part 1 submission
├── outputs/                      ← created when the notebook runs (git-ignored; force-add llm_ner_outputs.csv with `git add -f`)
└── situated-learning/
    └── outline.md                ← Part 2 skeleton → Assignment2_SituatedLearning_Group_81.pdf
```

## Dataset facts (verified)

CrossRE – AI domain (`DFKI-SLT/cross_re`, config `ai`): 100 train / 350 validation / 431 test sentences,
15 entity types in train+val (`algorithm, task, field, product, researcher, person, metrics, programlang,
conference, university, organisation, country, location, academicjournal, misc`; `event` appears only in test).
Span ends are **inclusive**. Top-5 relations in train+val (first label kept): `part-of, named, physical, related-to, role` → others `OTHER`.

## Plan – Part 1 (10 marks)

| Part | Marks | What the notebook does | Rubric "full marks" hooks |
| --- | --- | --- | --- |
| A Data + EDA | 1.0 | HF parquet load (GitHub fallback), inclusive→exclusive BIO, **round-trip check on every sentence**, counts/types/lengths, 3 printed examples, 10 % dev split of train+val for model selection | correct BIO verified on examples; complete EDA |
| B CRF | 1.5 | shape/case/affix/POS/±2 context features, c1/c2 grid + **feature-group ablation** on dev, per-type seqeval, top learned features | features justified, per-type report |
| C BiLSTM-CRF | 1.5 | GloVe 100d + char-CNN + BiLSTM + CRF, early stopping, **train vs val loss plot** | pretrained embeddings, curve, report |
| D Transformer | 2.0 | SciBERT cased, first-sub-token alignment (‑100), alignment table, epoch selection on dev, refit, per-type report | alignment correct **and explained** |
| E LLM NER | 1.5 | local `Qwen2.5-3B-Instruct`, zero-shot + 5-shot, JSON → spans, CSV of prompts/raw responses, parse-failure counts, latency | both settings, JSON mapped, outputs saved, failures counted |
| F Relation extraction | 1.5 | ordered gold pairs, top-5 + OTHER + NO_RELATION capped at 2×; LogReg baseline vs SciBERT with entity markers; macro-F1 + confusion matrices | two methods, macro-F1, CM, NO_RELATION handled |
| G Comparison | 1.0 | table built from variables on the same 100-sentence sample (+ strict F1, full-test F1, cost), per-type table, auto-categorised errors, 10 shown, auto-generated number summary | code-generated table, 10 errors, reflection matching numbers |

Integrity rules respected: seqeval entity-level metric, one test set, shared seeded sample for E/G, no mock LLM,
`SEED = 42` everywhere, numbers only from code.

## How to run (Colab, free T4)

1. Open `notebooks/Assignment2_Programming_Group_81.ipynb` in Colab (File → Upload, or open from GitHub).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all** (≈ 35–50 min; the LLM part is the longest).
4. File → Download `.ipynb` (executed, with outputs) and also download `outputs/llm_ner_outputs.csv`.
5. Commit both into this folder (`notebooks/` and `outputs/`).

## Checklist before submission

- [x] Group number, names and BITS IDs filled in; file renamed to Group_81.
- [x] Contribution column filled (equal contribution).
- [ ] Notebook executed top-to-bottom on Colab with **no error cells**.
- [ ] GloVe actually downloaded (Part C prints `GloVe 6B 100d: 400,000 vectors`, not the warning).
- [ ] Replace every `✏️ After running …` block with discussion that quotes the printed numbers.
- [ ] Reflection claims match the Part G table and the auto-generated summary.
- [ ] Part 2 PDF written from `situated-learning/outline.md`.

## Plan – Part 2 Situated learning (5 marks)

Separate 2–4 page PDF. Suggested topic: **8. Information Extraction** (fits our domain and Part 1 directly), or
3. RAG / 1. Open-source tools if that matches your workplace better. See `situated-learning/outline.md`.
It must describe a *real* use in a member's organisation/industry – or clearly say it is a proposed application.

## Workflow

1. ✅ Repo organised, brief + template stored, plan written.
2. ✅ Notebook drafted and smoke-tested end-to-end locally (tiny stand-in models, since the sandbox has no Hub access).
3. ⏳ Group runs the notebook on Colab and commits the executed copy.
4. ⏳ Write the result discussions (✏️ blocks) from the real numbers.
5. ⏳ Situated learning PDF.
