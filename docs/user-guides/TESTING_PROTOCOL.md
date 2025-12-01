# TESTING PROTOCOL

## 1. Mandate

This project will be defined by its reliability. Failure is not an option. A feature is not "done" until it is tested and passes all relevant checks outlined below. Adherence to this testing protocol is mandatory and non-negotiable.

## 2. Unit Testing

All core logic must be isolated and validated through unit tests. Use the `pytest` framework.

* **Target Modules:**
* **Hardware Profiler:** Your tests must mock different system configurations (`psutil`, `subprocess` calls) to ensure your VRAM/RAM detection and GPU layer calculation logic is sound under all conditions (e.g., no NVIDIA GPU, low RAM, high VRAM).
* **Prompt Router (Phase 2):** Create a suite of test prompts to verify that the routing logic correctly identifies the user's intent (e.g., "Write a python function..." correctly routes to the 'Coder' model).
* **RAG Retriever (Phase 3):** Test the document chunking and embedding logic. For the retrieval function, create a small, known FAISS index and test that specific queries return the expected document chunks.
* **Standard:** Every critical function must have a corresponding unit test.

## 3. Integration Testing

You will verify the end-to-end functionality of the entire system as components are completed.

* **Phase 1 Test:**
* **Scenario:** Execute the main script from the command line with a standard prompt.
* **Expected Outcome:** The script correctly logs the detected hardware, the calculated GPU layers, and prints a valid, complete response from the model to the console.

* **Phase 2 Test:**
* **Scenario:** Execute the script with a coding-related prompt, followed immediately by a general writing prompt.
* **Expected Outcome:** The application's logs must explicitly show the 'Coder' model being loaded for the first prompt, then unloaded, and the 'Writer' model being loaded for the second prompt. The test fails if both models are ever in memory simultaneously.

* **Phase 3 Test:**
* **Scenario:** Ingest a specific text document. Execute the script with the `--rag` flag and a prompt that can only be answered correctly with information from that document.
* **Expected Outcome:** The model's response correctly incorporates information from the ingested document.

## 4. Edge Case and Failure Mode Testing

A robust system anticipates failure. You must test for these scenarios:

* What happens if the `llama.cpp` binary is not found in the expected path?
* What happens if a model GGUF file is missing or corrupted?
* What happens if there is insufficient RAM/VRAM to load a model?
* How does the system respond to an empty or nonsensical prompt?

Your application must handle these errors gracefully and provide clear, informative error messages to the user.

---

No exceptions. No excuses.