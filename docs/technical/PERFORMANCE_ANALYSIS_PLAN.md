# PERFORMANCE ANALYSIS & OPTIMIZATION PLAN

## 1. Mandate

The system's viability is measured not just by its functionality, but by its performance. A slow tool is a useless tool. The currently benchmarked 2.7 TPS is unacceptable. This document outlines the mandatory, four-phase procedure for diagnosing, resolving, and validating the elimination of this performance bottleneck. Adherence to this plan is not optional.

---

## Phase 1: Baseline Establishment

**Objective:** Quantify the maximum possible performance of the Ollama backend in isolation to establish a non-negotiable performance target.

**Action Items:**
1.  Execute the `TinyLlama` model directly via the Ollama CLI, bypassing the AURA orchestrator entirely.
2.  Use a sufficiently long and simple prompt to ensure a period of sustained token generation. Example: `"Tell me a long story about a robot who discovered the meaning of art."`
3.  Record the raw **Tokens per Second (TPS)** reported by Ollama at the end of the generation. This metric is now the **Baseline TPS**.
4.  Create a new "Performance Analysis" section in your `BENCHMARKS.md` file and log this Baseline TPS.

---

## Phase 2: Overhead Isolation & Profiling

**Objective:** Identify the specific components within the Python orchestrator that are responsible for the latency between the Baseline TPS and the Application TPS.

**Action Items:**
1.  Run the exact same prompt from Phase 1 through your full `AURA-Engine-Core` application.
2.  Calculate the **Orchestrator Overhead** by subtracting the Application TPS from the Baseline TPS.
3.  Execute the application using Python's built-in profiler, `cProfile`, to generate a performance report.
    ```bash
    python -m cProfile -o profile.stats main.py --prompt "Your long prompt here..."
    ```
4.  Analyze the profiler output (`profile.stats`) and identify the top 5 functions with the highest cumulative execution time (`cumtime`).
5.  Document your findings—the Orchestrator Overhead and the top 5 slowest functions—in a new entry in your `OPERATIONAL_LOG.md`.

---

## Phase 3: Hypothesis & Refactoring

**Objective:** Implement a targeted solution to address the likely cause of the bottleneck.

**Primary Hypothesis:** The root cause of the latency is a synchronous, non-streaming HTTP request to the Ollama API. The orchestrator is waiting for the entire response to be generated before processing or displaying any output, crippling the perceived performance.

**Action Items:**
1.  Refactor the `OllamaWrapper` module to use a streaming API request. When using the `requests` library, this is achieved by setting `stream=True` in the API call.
2.  Modify the request logic to iterate over the response content as it arrives from the server.
3.  Modify the application's output handler to print each token to the console as it is received from the stream.
4.  Update the TPS calculation logic to function correctly with a streaming response.

---

## Phase 4: Validation

**Objective:** Empirically prove that the bottleneck has been eliminated and the system now performs at an acceptable standard.

**Action Items:**
1.  Re-run the same benchmark prompt from Phase 1 on your newly refactored application.
2.  Record the **New Application TPS**.
3.  Compare the New Application TPS to the Baseline TPS.
4.  Update the `BENCHMARKS.md` table with a new row for the "Optimized" application, showing the dramatic improvement.
5.  Update the performance numbers in your main `README.md`.

**Definition of Done:** This mission is complete only when the **New Application TPS** is within 10% of the **Baseline TPS** established in Phase 1.

---

A problem is not solved until the solution is measured and verified. Execute this plan precisely. No deviations.