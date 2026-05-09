# Thesis Reference Format Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the reference section format in the final thesis DOCX with the user-provided sample thesis document.

**Architecture:** Read the sample DOCX outside the workspace to extract the real paragraph formatting used by its reference heading and reference entries, then apply those exact paragraph-format settings to the current final thesis DOCX without changing reference content.

**Tech Stack:** Python 3.13, python-docx, PowerShell.

---

### Task 1: Inspect Sample and Current Reference Formatting

**Files:**
- Read: `C:\Users\??\?????\Downloads\??_??215_202112903529_???_??? .docx`
- Read: `??_??224-202212903403_??_??? .docx`

- [ ] **Step 1: Extract sample reference heading and entry paragraph format**
- [ ] **Step 2: Extract current thesis reference heading and entry paragraph format**
- [ ] **Step 3: Compare differences and lock the target formatting values**

### Task 2: Implement a Safe Reference-Format Fix Script

**Files:**
- Modify: `scripts/fix_final_docx.py`

- [ ] **Step 1: Add logic to locate the reference section in the final thesis DOCX**
- [ ] **Step 2: Apply sample-aligned formatting only to the reference heading and reference entry paragraphs**
- [ ] **Step 3: Keep the script content-safe so it does not rewrite bibliography text itself**

### Task 3: Apply the Fix to the Final Thesis DOCX

**Files:**
- Modify: `??_??224-202212903403_??_??? .docx`

- [ ] **Step 1: Run the fix script once**
- [ ] **Step 2: Re-open the final DOCX and verify the reference section formatting properties**
- [ ] **Step 3: Confirm that bibliography content remains unchanged**

### Task 4: Verification

**Files:**
- Verify: `scripts/fix_final_docx.py`
- Verify: `??_??224-202212903403_??_??? .docx`

- [ ] **Step 1: Run syntax verification**

Run: `python -m py_compile scripts/fix_final_docx.py`
Expected: exit code 0.

- [ ] **Step 2: Run the fix script**

Run: `python scripts/fix_final_docx.py`
Expected: target final DOCX path is printed without error.

- [ ] **Step 3: Re-read the final DOCX reference paragraphs and print the effective format values**

Expected: the current thesis reference section uses the same paragraph-format settings as the sample.
