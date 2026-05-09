# Thesis Chapter 5 Web System Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine Chapter 5 into a defense-ready web-system chapter, remove the PyQt5-focused subsection, and add a system architecture diagram plus a database relationship diagram that match the implemented FastAPI project.

**Architecture:** Keep the existing chapter skeleton but rewrite section 5.3 as a web-display-scheme section, strengthen 5.1/5.4/5.6/5.7 with code-backed descriptions, and add two generated figures tied to the actual modules in `app/`. Sync the revised Markdown back into the final thesis DOCX and verify that headings and figure captions are present.

**Tech Stack:** Markdown, python-docx, Pillow, FastAPI project sources, PowerShell, Python 3.13.

---

### Task 1: Revise Chapter 5 Markdown Content

**Files:**
- Modify: `docs/paper/graduation_final_draft.md`

- [ ] **Step 1: Rewrite the Chapter 5 structure around the web system**

Change the section wording so that `5.3` no longer mentions PyQt5 and instead focuses on the rationale and structure of the web display scheme. Strengthen `5.1`, `5.4`, `5.6`, and `5.7` so they read like thesis prose rather than implementation notes.

- [ ] **Step 2: Add explicit figure references in the prose**

Insert references for `?5-2 ???????` in section `5.1` and `?5-3 ????????` in section `5.6`, making sure each figure is introduced before it is inserted into the DOCX.

- [ ] **Step 3: Keep the content aligned with actual code**

Use the real modules `app/web_demo.py`, `app/utils/web_inference.py`, `app/db.py`, `app/db_models.py`, `app/config.py`, and `scripts/system_smoke_test.py` as the basis for all text so the chapter remains defensible.

### Task 2: Generate Thesis Figures for Chapter 5

**Files:**
- Create: `scripts/generate_thesis_ch5_diagrams.py`
- Create: `figures/?13_???????.png`
- Create: `figures/?14_????????.png`

- [ ] **Step 1: Generate a system architecture figure**

Create a PNG figure showing the relation among the browser UI, FastAPI routes, `DetectionService`, YOLO weights, runtime outputs, and the database/file storage split.

- [ ] **Step 2: Generate a database relationship figure**

Create a PNG figure showing `model_registry`, `app_settings`, `image_detections`, `image_detection_boxes`, and `video_tasks`, including the main foreign-key relations used in the project.

- [ ] **Step 3: Keep diagram generation repeatable**

Use a Python script so the figures can be regenerated if the thesis layout changes later.

### Task 3: Sync Figures into the Final Thesis DOCX

**Files:**
- Modify: `scripts/sync_final_docx_from_markdown.py`
- Modify: `??_??224-202212903403_??_??? .docx`

- [ ] **Step 1: Extend the DOCX sync image plan**

Register `?13_???????.png` after section `5.1` and `?14_????????.png` after section `5.6`, while preserving the existing Chapter 2/4/5 image insertion logic.

- [ ] **Step 2: Run the sync command**

Run: `python scripts/sync_final_docx_from_markdown.py`
Expected: the final thesis DOCX is updated successfully and a backup is saved under `output/doc/`.

- [ ] **Step 3: Verify the updated DOCX content**

Check that the final DOCX contains the revised `5.3` heading, the two new Chapter 5 figure captions, and the expected number of inline images.

### Task 4: Run Minimal Verification

**Files:**
- Verify: `docs/paper/graduation_final_draft.md`
- Verify: `scripts/sync_final_docx_from_markdown.py`
- Verify: `??_??224-202212903403_??_??? .docx`

- [ ] **Step 1: Verify script syntax**

Run: `python -m py_compile scripts/sync_final_docx_from_markdown.py scripts/generate_thesis_ch5_diagrams.py`
Expected: exit code 0.

- [ ] **Step 2: Regenerate and sync artifacts**

Run: `python scripts/generate_thesis_ch5_diagrams.py` and `python scripts/sync_final_docx_from_markdown.py`
Expected: both commands finish without errors.

- [ ] **Step 3: Verify headings and captions in the final thesis**

Export the final DOCX paragraph text and check that `5.3 ????????`, `?5-2 ???????`, and `?5-3 ????????` are present.
