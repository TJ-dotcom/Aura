# MODEL SELECTION RATIONALE (v2)

## 1. Mandate

Model selection is a **dynamic, multi-tiered strategy**. The AURA Engine will programmatically select the most capable model that fits within the user's detected hardware constraints. This document outlines the official tiered model roster. The system's primary intelligence lies in its ability to navigate these tiers automatically, creating a truly adaptive experience.

## 2. Guiding Principles

1.  **Tiered Deployment:** We will define multiple performance tiers (e.g., High-Performance, Balanced, High-Efficiency). The system's first action on startup will be to profile the user's hardware and select the highest possible tier it can support.
2.  **Resource-First Selection:** Within each category, the chosen model is determined by the active performance tier. The goal is to always use the strongest model the user's hardware can comfortably accommodate.
3.  **Mandatory Quantization:** All models will be used in a 4-bit quantized GGUF format (e.g., Q4_K_M) to maximize efficiency across all tiers.

## 3. Tiered Model Roster & Hardware Thresholds

The following tables define the model selection logic. The Hardware Profiler must be programmed to determine the active tier based on these VRAM thresholds.

### Category: Text/Reasoning

| Tier          | Model               | Parameters | Min. VRAM Req. |
|---------------|---------------------|------------|----------------|
| **Tier 1 (High-Perf)** | DeepSeek-R1-D 8B    | 8B         | >10 GB         |
| **Tier 2 (Balanced)** | Orca-2              | 7B         | >8 GB          |
| **Tier 3 (High-Eff)** | Phi-3.5-mini        | 3.8B       | <8 GB          |
**

### Category: Coding

| Tier          | Model               | Parameters | Min. VRAM Req. |
|---------------|---------------------|------------|----------------|
| **Tier 1 (High-Perf)** | DeepSeek-R1-D 8B    | 8B         | >10 GB         |
| **Tier 2 (Balanced)** | DeepSeek-R1-D 7B    | 7B         | >8 GB          |
| **Tier 3 (High-Eff)** | Qwen2               | 7B         | >8 GB          |
**

*Note: Tier 2 and 3 for Coding have similar requirements. The system should prioritize the higher-ranked model (`DeepSeek-R1-D 7B`) when VRAM allows.*

### Category: Mathematics

| Tier          | Model               | Parameters | Min. VRAM Req. |
|---------------|---------------------|------------|----------------|
| **Tier 1 (High-Perf)** | DeepSeek-R1-D 7B    | 7B         | >8 GB          |
| **Tier 2 (Balanced)** | DeepSeek-R1-D 8B    | 8B         | >10 GB         |
| **Tier 3 (High-Eff)** | DeepSeek-R1-D 1.5B  | 1.5B       | <4 GB          |
**

*Note: The ranking for Mathematics is distinct. The 7B model is ranked higher than the 8B for this specific task.*

---
## 4. Architectural Directive: Document Parsing & RAG

The tiered selection logic **does not** apply to the Document Parsing models listed in the source image.

* **Clarification:** Models like `DistilBERT` and `TinyBERT` are **encoder-only** models. They are architecturally unsuited for generative tasks within our system. Their role is specialized.
* **Implementation:** A single, lightweight sentence-transformer model (e.g., one based on `DistilBERT`) will be used for the **embedding/ingestion phase** of the RAG pipeline. This is a separate, non-negotiable choice that is independent of the performance tiers.
* The final **synthesis** of retrieved documents will be handled by the active **Text/Reasoning** model from the selected tier.

## 5. Implementation Mandate

The Hardware Profiler must be upgraded. Its first job is to determine and set a global "Performance Tier" for the session. The Orchestrator will then query this tier to select the correct model for download and execution. The user must be notified of their active performance tier on startup.

This is the new plan. The complexity has increased, but so has the potential of the project. Execute.