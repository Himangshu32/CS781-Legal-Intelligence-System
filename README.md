# CS781 Capstone Project: In-House Legal Intelligence System

**Student Name:** Himangshu Mahato  
**Roll Number:** 261110027  
**Email:** `himangshum26@iitk.ac.in`  
**Institution:** Indian Institute of Technology Kanpur (IIT Kanpur)  
**Codabench Username:** `h_261110027`  
**GitHub Repository:** [https://github.com/Himangshu32/CS781-Legal-Intelligence-System](https://github.com/Himangshu32/CS781-Legal-Intelligence-System)

---

## Abstract & System Overview

This repository contains the complete codebase, experimental logs, preprocessing scripts, and final report for the **CS781 Capstone Project**. The objective of this project was to design, pretrain, and evaluate an entirely self-hosted, confidential legal Natural Language Processing (NLP) pipeline built completely from scratch without relying on external pretrained foundation models or closed commercial APIs.

To meet the multi-task legal intelligence challenge under strict parameter constraints (50M–300M parameter limit), a custom **101M-parameter Grouped-Query Attention (GQA)** causal foundation model was pretrained on an **18.5 GB** deduplicated corpus of Indian legal texts (Acts, Supreme Court, High Courts, and Tribunals). This shared causal backbone was subsequently transferred across **seven distinct phases** using advanced Parameter-Efficient Fine-Tuning (PEFT) techniques, task-specific decoding strategies, and hybrid retrieval mechanisms.

---

## Repository Structure

```text
CS781-Legal-Intelligence-System/
│
├── code/
│   ├── phase1/          # Phase 1: Legal LM Pretraining & Custom Tokenizer
│   ├── phase2/          # Phase 2: Legal Named Entity Recognition (L-NER)
│   ├── phase3/          # Phase 3: Rhetorical Role Prediction
│   ├── phase4/          # Phase 4: Court Judgment Outcome Prediction (CJPE)
│   ├── phase5/          # Phase 5: Legal Statute Identification (LSI)
│   ├── phase6/          # Phase 6: Abstractive Summarization
│   └── phase7/          # Phase 7: Prior Case Retrieval (PCR) & Hybrid Reranking
│
├── report/
│   └── CS781-CP-Himangshu-Mahato-261110027-report.pdf  # Final compiled academic report
│
└── README.md            # Comprehensive project documentation
