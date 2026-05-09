# Thesis Page Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the thesis body by enriching Chapter 4 and Chapter 5 without materially increasing AI-style writing risk.

**Architecture:** Use a scripted DOCX rewrite so the new content can be inserted at controlled positions in the existing thesis. Add project-specific experimental discussion in Chapter 4 and implementation-specific system detail in Chapter 5, then re-run the existing formatting alignment script.

**Tech Stack:** `python-docx`, local rewrite scripts, existing thesis format alignment script.

---

### Task 1: Inspect current chapter structure

**Files:**
- Read: `定稿_智能224-202212903403_陈刚_林一锋 .docx`

- [ ] Export the current paragraph index map for Chapter 4 and Chapter 5.
- [ ] Identify safe insertion points after existing topic paragraphs and before the next heading.
- [ ] Freeze the insertion targets so the rewrite stays scoped to the two chapters.

### Task 2: Draft low-risk expansion text

**Files:**
- Create: `scripts/rewrite_thesis_page_expansion.py`

- [ ] Write Chapter 4 additions focused on small-object difficulty, error cases, model comparison interpretation, and experiment boundary discussion.
- [ ] Write Chapter 5 additions focused on architecture layering, service flow, database use, stability handling, and system testing.
- [ ] Keep every new paragraph tied to concrete project terms and avoid generic filler phrasing.

### Task 3: Apply the chapter expansion

**Files:**
- Modify: `定稿_智能224-202212903403_陈刚_林一锋 .docx`
- Modify: `output/doc/thesis_sample_aligned.docx`

- [ ] Back up the thesis before writing.
- [ ] Insert new paragraphs into the target sections while preserving heading styles.
- [ ] Sync the aligned thesis copy after the rewrite.

### Task 4: Re-align format and verify

**Files:**
- Modify: `定稿_智能224-202212903403_陈刚_林一锋 .docx`
- Read: `定稿_智能224-202212903403_陈刚_林一锋 .docx`

- [ ] Run the thesis format alignment script in place.
- [ ] Re-open the DOCX and sample-check the inserted paragraphs.
- [ ] Confirm that headings remain intact and that the added text reads like project-specific analysis rather than generic expansion.

