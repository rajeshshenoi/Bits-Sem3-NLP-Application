---
title: "Assignment 2 – Part 2: Situated Learning"
subtitle: "Topic 7 – Machine Translation in the Automotive Industry"
---

**AIML ZG519 – Natural Language Processing Applications · Group 81**

## Student details

| | |
| --- | --- |
| **Team member names** | RAJESH R. SHENOI (2025AE05200), PARUL MITTAL (2025AE05198), SIDDHARTH KUCHIMANCHI (2025AE05199), CHAUHAN AYUSH UPENDRA (2025AE05684) |
| **BITS Email id** | ⟨email 1⟩, ⟨email 2⟩, ⟨email 3⟩, ⟨email 4⟩ |
| **Industry / Domain** | Automotive |
| **Organisation type** | ⟨confirm, e.g. automotive supplier / engineering organisation serving vehicle manufacturers⟩ – workplace of group member Rajesh R. Shenoi, anonymised |

> **How to read this report.** Sections marked **[Observed]** describe how machine translation is used in the
> organisation today, as reported by the group member who works there. Sections marked **[Proposed]** are the
> group’s own suggestion. No confidential, customer or personal information is included; all example sentences are
> synthetic.

## 5.1 Industry context – Anchor

| | |
| --- | --- |
| **Industry / Organisation** | Automotive; an organisation that receives technical documents from vehicle‑manufacturer customers in Germany and Japan. |
| **Selected NLP concept** | Machine Translation (MT) – neural, Transformer‑based MT, and LLM‑based translation. |
| **Application / use case** | **[Observed]** Translating German and Japanese customer documents – e.g. **customer‑specific requirements (CSRs)** and **vehicle specifications** – into English, so that engineering, quality and project teams can understand and implement them. |
| **Problem addressed** | Customers write requirements in their own language, while the engineering teams work in English. Every requirement that is misunderstood can lead to a design error, a failed validation or a contractual non‑conformity. Human translation of long specifications is slow and expensive; MT gives engineers fast access to the content. |
| **Users / stakeholders** | Engineers and project managers (readers of the translation); the **language‑competent department** that reviews translations; quality/requirements managers; the customer (owner of the document and of its confidentiality); IT/security and legal (approve tools and enforce confidentiality agreements). |
| **Status** | **Deployed** – MT is used today for this purpose. |

## 5.2 Concept in practice – Apply

**Data / input.** **[Observed]** German and Japanese technical documents received from customers: customer‑specific
requirements and vehicle specifications. Typical content (illustrated with synthetic sentences in the Evidence
section): obligations (“*muss*”, “*しなければならない*” → *shall*), numeric limits with units (temperatures, torque,
time limits), references to standards (e.g. ISO 26262), and dense domain terminology (*Steuergerät* – control unit,
*Lastenheft* – requirements specification, *耐久試験* – durability test).

**NLP approach / concept (course link).** Modern MT systems are **neural encoder–decoder Transformers**: the encoder
builds contextual representations of the source sentence with self‑attention, and the decoder generates the target
sentence token by token while attending to the source (cross‑attention). Text is split into **sub‑word units**
(SentencePiece/BPE), so rare compounds such as *Dauerlaufprüfung* can still be translated, and decoding uses **beam
search**. **Multilingual models** (e.g. NLLB‑200) share one model across many language pairs; **LLMs** translate by
prompting. Quality is measured with **BLEU** and **chrF** against reference translations, and in practice with human
**post‑editing (MTPE)**.

**Tool / model / framework.** **[Observed]** Google Translate and other translation tools approved by the enterprise.
**[Proposed – demo]** Open‑source models that run locally: Helsinki‑NLP **OPUS‑MT** (Marian, de→en and ja→en) and
Meta **NLLB‑200 distilled 600M**, run with HuggingFace `transformers` on a CPU.

**Workflow / architecture.**

![Current and proposed MT workflow](mt_workflow.png)

* **[Observed] current workflow:** document received → text translated with Google Translate or an approved tool →
  output **used as is** for understanding, or **reviewed by the language‑competent department** when accuracy matters.
  Because of the customers’ **confidentiality agreements**, documents may not be put into the public domain, so
  public/commercial web translators and API‑ or LLM‑based translation services cannot be used for these documents.
* **[Proposed] on‑premise workflow:** document stays inside the company network → pre‑processing (split into
  sentences, protect numbers/units/standard names) → **locally hosted open‑source MT model** with a **domain
  glossary** → automatic checks (numbers, glossary terms, obligations) → low‑confidence sentences routed to the
  language department for post‑editing → approved translation stored in a translation memory, which later improves
  the model.

**Output / result.** **[Observed]** English versions of the customer documents used by the engineering teams, either
raw or after review. **[Proposed – demo]** See the evidence table below: quality scores, terminology accuracy and CPU
speed of two open‑source models on synthetic automotive requirement sentences, with no data leaving the machine.

**Evidence.**

*E1 – Demo results (synthetic data, open‑source models, offline CPU inference; notebook `mt_demo.ipynb`).*

⟨DEMO TABLE – filled from the executed notebook: model, language pair, BLEU, chrF, glossary terms correct,
numbers/units preserved, CPU seconds per sentence⟩

*E2 – Example input → output (synthetic sentence).*

⟨DEMO EXAMPLES – 2–3 rows: German/Japanese source, MT output, reference⟩

*E3 – Workflow diagram* (above). Screenshots of the internal tools are not included because of confidentiality.

## 5.3 Industry analysis – Analyse

**Key benefits / value.**

* **Speed and access:** engineers can read a German or Japanese specification within minutes instead of waiting for
  a full human translation – important when requirements arrive during quotations or design changes.
* **Cost:** human translation is reserved for the documents and passages that need it (review by the language
  department), instead of every page.
* **Coverage:** one workflow covers both German and Japanese customers, and multilingual models can add further
  languages.

**Limitations / problems.**

* **Confidentiality (major):** **[Observed]** customer agreements forbid uploading documents to public domains, which
  rules out public web translators, cloud MT APIs and LLM services for many documents – exactly the documents where
  translation is most needed.
* **Domain terminology:** general‑purpose MT does not know company or customer terminology (*Lastenheft*,
  *Bordnetz*, abbreviations); one wrong term can change a requirement.
* **Requirement semantics:** modal verbs and negations (*muss* = *shall*, *darf nicht* = *must not*), numbers and units
  must be translated exactly; fluent but wrong output is dangerous because it looks correct.
* **Japanese is harder:** long sentences, omitted subjects and different word order give lower quality than
  German→English (see E1).
* **Raw output risk:** **[Observed]** output is sometimes used as is, without review, so errors may go unnoticed.

**Evidence of effectiveness or impact.** **[Observed]** MT is part of the daily workflow for German and Japanese
documents, which shows that it is useful enough to be adopted, with the language department as the quality gate.
**[Proposed – demo]** ⟨one or two sentences quoting E1, e.g. which model reaches which chrF, how many glossary
terms/numbers were kept, and the CPU time per sentence – showing that local models are fast enough for on‑premise
use but still need a glossary and review⟩.

## 5.4 Reflection and future scope – Reflect

**Major challenge / gap.** The **confidentiality constraint**: the best‑known MT and LLM services are cloud‑based, but
customer confidentiality agreements do not allow these documents to leave the organisation. The current tools
therefore cannot be used freely for the documents that matter most, and translation either falls back to slower
manual work or carries a compliance risk.

**Recommended improvement.** **[Proposed]** An **on‑premise MT service**: open‑source Transformer models (e.g.
OPUS‑MT, NLLB‑200, or a larger open LLM where hardware allows) hosted on an internal server, with (1) a **domain
glossary** enforced during or after translation, (2) automatic **checks** for numbers, units, standard names and
obligation/negation words, (3) **routing** of low‑confidence sentences to the language department, and (4)
**fine‑tuning** on the organisation’s own reviewed translations (translation memory). Requirements: approval from IT
security and legal, an internal server (the demo shows a CPU is sufficient for small models; a GPU helps for larger
ones), a glossary owner, and a small test set to measure quality before roll‑out.

**Potential future use.** Translate‑and‑compare of **new customer document versions** (highlight changed
requirements), **multilingual requirement search/RAG** over historic specifications, extraction of requirements into
the requirements‑management tool after translation (link to Information Extraction), and translation of supplier
or plant communication in further languages.

## References and acknowledgements

* Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
* Tiedemann, J. & Thottingal, S. (2020). *OPUS‑MT – Building open translation services for the World.* EAMT.
* NLLB Team (2022). *No Language Left Behind: Scaling Human‑Centered Machine Translation.* arXiv:2207.04672.
* Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores* (sacreBLEU). WMT.
* Popović, M. (2015). *chrF: character n‑gram F‑score for automatic MT evaluation.* WMT.
* Tools: HuggingFace `transformers`, `sacrebleu`; Google Colab for the demo.
* Information on the organisational practice provided by group member Rajesh R. Shenoi (anonymised).
* AI tools: Claude Code (Anthropic) was used to help structure the report, write the demo notebook and prepare the
  synthetic example sentences and references; the content was reviewed by the group.
