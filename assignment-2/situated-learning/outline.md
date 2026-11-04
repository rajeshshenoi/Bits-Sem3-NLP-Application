# Part 2 – Situated Learning (5 marks) – Topic 7: Machine Translation

Deliverable: `Assignment2_SituatedLearning_Group_81.pdf`, 2–4 pages (screenshots, sample outputs and evidence
not counted). Flow required: **CONCEPT TAUGHT → INDUSTRY USE → PRACTICAL ANALYSIS → REFLECTION**.

## Ground rules from the brief (what the evaluator checks)

* It must describe how MT **is used, is being explored, or could realistically be used** in a member's real
  industry/organisation (§3). If it is not used, say so plainly and propose a realistic application (§6) – never
  invent a deployment.
* Clearly separate **observed practice** from **our own proposal** (§9). Label each part.
* No confidential, proprietary or personal data: anonymise the organisation, use synthetic sample text (§9).
* Evidence carries the 2-mark "Apply" section: anonymised screenshots, sample input → output, an architecture
  diagram, tool documentation (§7).
* Acknowledge every tool, model, dataset and source (and AI tools) used.

## Facts we need from the group (cannot be invented)

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which member's workplace? Industry and organisation type (anonymised) | |
| 2 | Is MT used there? deployed / pilot / not used (→ proposed) | |
| 3 | What gets translated? (tickets, emails, documentation, UI strings, contracts, chat, code comments …) language pairs, rough volume | |
| 4 | Which tool? (Google/Microsoft/DeepL/AWS Translate, an LLM, open-source NLLB/MarianMT, a translation vendor, none) | |
| 5 | Who uses the output and how? (raw MT, human post-editing, review step) | |
| 6 | Known problems? (domain terms, quality, confidentiality, cost, latency) | |
| 7 | Any evidence we can show? (anonymised screenshot, sample, workflow sketch) | |

## Section plan with MT-specific content

### Student details
Team member names, BITS email ids, Industry / Domain, Organisation type.

### 5.1 Industry context – Anchor [1 mark]
* Industry / organisation (anonymised), selected concept: **Machine Translation (neural MT / LLM-based MT)**.
* Application / use case – e.g. translating customer-support tickets, localising product documentation, translating
  internal knowledge-base articles for offshore teams.
* Problem addressed – multilingual users/clients vs. English-only (or regional-language) teams; slow, costly human
  translation; turnaround for support SLAs.
* Users / stakeholders – support agents, end customers, technical writers, localisation team, compliance/legal.
* Status: deployed / pilot / proposed (state explicitly).

### 5.2 Concept in practice – Apply [2 marks]
* **Data / input:** document types, formats (text, HTML, PDF, chat), language pairs, volume.
* **NLP approach:** concept taught – encoder–decoder Transformer NMT (attention, subword vocabularies), multilingual
  models, LLM prompting for translation; supporting pieces: glossary/terminology enforcement, translation memory,
  quality estimation, human post-editing (MTPE).
* **Tool / model / framework:** the actual tool used (or proposed: e.g. open-source MarianMT / NLLB-200 via
  HuggingFace for on-premise privacy, or a cloud MT API).
* **Workflow / architecture:** ingest → pre-process (segment, protect code/placeholders, PII masking) →
  MT engine (+ glossary) → quality estimation → human post-edit if low score → publish/store → feedback loop.
  Include a diagram.
* **Output / result:** translated text, turnaround, any quality metric (BLEU/chrF/COMET on a sample, or
  post-editing effort).
* **Evidence:** anonymised screenshot or a **small synthetic demo** (e.g. 5 representative sentences translated with
  an open-source model, scored with chrF/BLEU against a human reference) – clearly labelled as a demonstration.

### 5.3 Industry analysis – Analyse [1 mark]
* Benefits: speed, cost per word, 24×7 multilingual coverage, consistency with glossaries.
* Limitations: domain terminology, low-resource languages, mistranslated negation/numbers, formality/tone,
  hallucination (LLMs), confidentiality of text sent to cloud APIs, metric–quality gap (BLEU vs. human judgement).
* Effectiveness / impact: observed numbers if available (turnaround, volume, post-edit rate), otherwise the demo
  results and published benchmarks – labelled as such.

### 5.4 Reflection and future scope – Reflect [1 mark]
* One major challenge (e.g. domain terminology / confidentiality / quality assurance).
* One practical improvement (e.g. glossary-constrained MT + quality-estimation routing; fine-tuning on the
  organisation's translation memory; on-premise open model).
* Potential future use (e.g. speech-to-speech for support calls, multilingual RAG/search).

### References / acknowledgements
Tools, models, datasets, papers (e.g. Vaswani et al. 2017; NLLB Team 2022), any AI tools used.
