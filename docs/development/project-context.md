# PROJECT MANDATE: AURA-ENGINE-CORE

## 1. Mission Briefing

This document outlines the non-negotiable development plan for the AURA-Engine-Core project. This is not a product development exercise. It is a strategic operation to forge a portfolio piece that demonstrates elite engineering capabilities in AI systems.

Our evaluation of success is not a functional UI or a large user base. It is the demonstrable quality, robustness, and efficiency of the backend systems you create. Every line of code must serve this primary objective.

The guiding principles are as follows:
* **Execution over Ideation:** We are past the ideas stage. The plan is set.
* **Substance over Style:** The core logic is all that matters. Aesthetics are a distraction.
* **Backend over Everything:** The interface is irrelevant until the engine is perfect.

## 2. Rules of Engagement

1.  **CLI First:** All development and testing will be done through a Command-Line Interface. No GUI or web development is authorized until all backend phases are complete and validated.
2.  **Focus on the Hard Problems:** Your time will be spent on memory management, system profiling, I/O, and pipeline efficiency—not on superficial features.
3.  **Develop in Stealth:** This project is built in private. Public announcements or posts will be made only after a major phase is complete, polished, and approved. Control the narrative.
4.  **Maintain Architecture Documentation:** The complete codebase structure MUST be tracked and updated in the README.md Architecture section after each phase completion. This serves as both documentation and a living record of system evolution for portfolio demonstrations.

## 3. Phased Execution Plan

### Phase 1: The Hardware-Aware Inference Core
* **Objective:** Prove low-level system resource management.
* **Critical Path:**
1.  Build a Python script utilizing `argparse` for CLI prompt input.
2.  Implement robust hardware detection for system RAM and GPU VRAM.
3.  Develop logic to calculate the optimal `--n-gpu-layers` setting for `llama.cpp`.
4.  Use `subprocess` to call the `llama.cpp` binary with the dynamically generated settings.
* **Definition of Done:** A script that accepts a prompt via CLI, correctly profiles hardware, loads a model with appropriate GPU layers, runs inference, and prints all diagnostics and results to the console without error.

### Phase 2: The Dynamic Model Orchestrator
* **Objective:** Engineer a resource-conscious system for switching between specialized models.
* **Critical Path:**
1.  Refactor the Phase 1 script into a main `Orchestrator` class.
2.  Implement a rule-based `Router` to select a model based on prompt keywords.
3.  The Orchestrator MUST unload the current model from memory completely before loading the new one.
4.  Add verbose console logging to track all model loading/unloading events.
* **Definition of Done:** The application can receive two different types of prompts consecutively and correctly use a different specialized model for each, while demonstrably keeping only one model in memory at a time.

### Phase 3: The RAG Integration
* **Objective:** Augment the core engine with a local retrieval pipeline.
* **Critical Path:**
1.  Integrate the FAISS library.
2.  Build a separate ingestion script to create a local vector index from documents.
3.  Modify the main script to accept a `--rag` flag.
4.  When flagged, the system must retrieve relevant context from the FAISS index and prepend it to the prompt before sending it to the LLM.
* **Definition of Done:** The CLI tool can successfully answer questions that require knowledge from a user-provided document, which it accesses through its local RAG pipeline.

---

No deviation from this plan is authorized. Report upon completion of each phase.

**Execute.**