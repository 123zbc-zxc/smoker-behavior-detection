# Reduce AIGC Risk in Thesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lower the paper's reported AIGC risk by rewriting the most heavily flagged sections without changing project facts, experiment metrics, figures, or thesis structure.

**Architecture:** Work directly on the thesis DOCX through scripted paragraph replacement. Prioritize the English abstract, Chapter 2, Chapter 5, and Chapter 6, then revise selected explanatory paragraphs in Chapters 3 and 4 that match the report's flagged fragments.

**Tech Stack:** `python-docx`, local HTML report parsing, existing thesis formatting scripts.

---

### Task 1: Locate flagged sections

**Files:**
- Read: `tmp/ai_report_unzip/定稿_智能224-202212903403_陈刚_林一锋_AIGC统计报告.html`
- Read/Modify: `定稿_智能224-202212903403_陈刚_林一锋 .docx`

- [ ] Confirm chapter-level AIGC ratios and identify the highest-risk chapters.
- [ ] Map the flagged report fragments back to the actual paragraph ranges in the thesis.
- [ ] Freeze the ranges to be rewritten so later edits do not spill into unaffected sections.

### Task 2: Draft lower-risk replacement text

**Files:**
- Modify: `定稿_智能224-202212903403_陈刚_林一锋 .docx`

- [ ] Rewrite the English abstract into shorter, less template-like academic prose.
- [ ] Rewrite Chapter 2 into more concrete, course-paper style technical exposition.
- [ ] Rewrite Chapter 5 into implementation-centered language tied to this project's real code and deployment choices.
- [ ] Rewrite Chapter 6 into restrained concluding language with fewer generic outlook phrases.
- [ ] Rewrite only the specifically flagged explanatory paragraphs in Chapters 3 and 4.

### Task 3: Apply the changes safely

**Files:**
- Modify: `定稿_智能224-202212903403_陈刚_林一锋 .docx`
- Modify: `output/doc/thesis_sample_aligned.docx`

- [ ] Create a fresh backup before writing new content.
- [ ] Replace paragraph text while preserving existing heading levels and document formatting.
- [ ] Save both the main thesis file and the aligned copy.

### Task 4: Verify content and formatting

**Files:**
- Read: `定稿_智能224-202212903403_陈刚_林一锋 .docx`

- [ ] Re-open the DOCX and verify the rewritten sections are present.
- [ ] Check that title hierarchy, references, and table formatting still remain intact.
- [ ] Summarize which sections were rewritten and note any remaining likely high-risk areas for a second pass.
