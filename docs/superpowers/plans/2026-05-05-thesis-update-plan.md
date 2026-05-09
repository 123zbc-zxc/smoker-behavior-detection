# Thesis Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a current-project DOCX thesis based on the existing thesis and verified project evidence.

**Architecture:** Use the original DOCX as content/template reference, generate a clean new DOCX with python-docx, update thesis narrative to emphasize dataset construction, baseline/ECA/SE comparison, hard fine-tune model selection, TTA+SAHI offline enhancement, and FastAPI system implementation. Verify the resulting DOCX by rendering pages to PNG.

**Tech Stack:** python-docx, repository JSON/Markdown reports, document render script from the documents skill.

---

### Task 1: Evidence Mapping

**Files:**
- Read: `定稿_智能224-202212903403_陈刚_林一锋 .docx`
- Read: `README.md`
- Read: `runs/val/*/test_summary.json`
- Read: `runs/reports/final_model_system_test_report_20260503.json`
- Read: `output/enhanced_eval/verify_sample50.json`

- [x] Confirm current thesis structure and tables.
- [x] Confirm baseline/ECA/SE/hard fine-tune/TTA+SAHI metrics.

### Task 2: Generate DOCX

**Files:**
- Create: `定稿_智能224-202212903403_陈刚_林一锋_项目最新版.docx`

- [ ] Build a full thesis with cover, declarations, abstracts, table of contents, chapters, references, and appendix.
- [ ] Use verified numbers only; mark limitations carefully.
- [ ] Keep Chinese thesis style and concise academic tone.

### Task 3: Render QA

**Files:**
- Read: `定稿_智能224-202212903403_陈刚_林一锋_项目最新版.docx`
- Create: `tmp/docx_render_thesis_latest/`

- [ ] Render the DOCX to page PNGs.
- [ ] Inspect rendered page list and representative pages.
- [ ] Fix obvious layout or rendering failures before delivery.
