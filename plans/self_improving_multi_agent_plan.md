# Self-Improving Multi-Agentic Data Analysis System

## System and Implementation Plan

---

## 1. System Task

The system takes a CSV/Excel file and a natural language question about it as input. It produces a structured response: textual analysis, numerical results, and a chart if relevant.

Example questions:

- *"Which region had the largest revenue growth in 2023 compared to the previous year?"*
- *"Is there a significant difference between the average basket value of male and female customers?"*
- *"Show the monthly revenue trend by product category, and describe what you see."*
- *"What are the top 5 most frequent error types, and is there a correlation between them and the month?"*

This task is a good choice because question complexity varies, so the system has to adaptively decide on the number of steps and the tool usage — exactly the area where the self-improving component has room to optimize.

---

## 2. Agents

Five agents. For each one I specify the responsibility, I/O, system prompt outline, tools used, and the model.

### 2.1 Planner

**Responsibility**: decomposes the user's question into executable substeps and defines the workflow structure. Decides whether a chart is needed, whether a statistical test is needed, whether a textual summary is needed.

**Input**: the user question + the CSV schema (from `schema_inspector`).

**Output**: a structured JSON plan:

```json
{
  "steps": [
    {"id": 1, "agent": "DataAnalyst", "task": "Compute yearly revenue growth by region", "depends_on": []},
    {"id": 2, "agent": "Visualizer", "task": "Bar chart of growth per region", "depends_on": [1]},
    {"id": 3, "agent": "Critic", "task": "Verify the calculation", "depends_on": [1]},
    {"id": 4, "agent": "Writer", "task": "Summarize the regional trends", "depends_on": [1, 2]}
  ],
  "expected_output_type": "report_with_chart"
}
```

**System prompt outline**: *"You plan data analysis tasks. Given a CSV schema and a user question, decompose the task into a minimal number of substeps. Don't add unnecessary steps — if no chart or text summary is needed, don't request a Visualizer or Writer. Output: strict JSON."*

**Tool**: `schema_inspector`. Does not run code.

**Model**: Sonnet-tier (Claude Sonnet 4.6 or GPT-4.1).

### 2.2 DataAnalyst

**Responsibility**: writes Python code (pandas/numpy/scipy) and runs it. Performs the actual computation. Up to 3 iterations per task (code → error → fix).

**Input**: the task from the Planner + path to the CSV + results from previous steps.

**Output**:

```json
{
  "code": "import pandas as pd\n...",
  "result": {"type": "dataframe", "data": [...], "summary": "..."},
  "stdout": "...",
  "success": true
}
```

**System prompt outline**: *"You are a pandas expert. You receive a concrete subtask and a CSV. Write short, correct Python code. Print the result in a structured way. If you get an error, fix it. Maximum 3 attempts."*

**Tools**: `python_sandbox`, `sql_query`, `statistical_test`, `schema_inspector`.

**Model**: Sonnet or Haiku. Haiku is also good for pandas code and is cheaper — worth measuring.

### 2.3 Visualizer

**Responsibility**: generates a chart from the DataAnalyst's result. Only invoked if the Planner requested it.

**Input**: the visualization description from the Planner + the DataAnalyst's output.

**Output**: path to the PNG file + short caption.

**System prompt outline**: *"You are a data visualization expert. You write code that generates a clear, well-labeled chart. Always include a title, axis labels, and pick a chart type that fits the question."*

**Tools**: `chart_generator`, `python_sandbox` (fallback for complex charts), `file_io`.

**Model**: Haiku.

### 2.4 Critic

**Responsibility**: reviews the output of DataAnalyst and Visualizer. At two levels:

- **Code-level**: logic errors (wrong column, missing NaN handling, off-by-one).
- **Result-level**: is the result consistent with the question (negative revenue, implausible numbers).

**Input**: the original question + the Planner's plan + DataAnalyst's code and result + Visualizer's output.

**Output**:

```json
{
  "verdict": "approved" | "needs_revision",
  "issues": ["You computed 2023 revenue, but the question asks about growth"],
  "suggested_action": "rerun_data_analyst" | "rerun_planner" | null
}
```

**System prompt outline**: *"You are a critical analyst. You look for concrete, technical errors. If everything is fine, say 'approved'. Only flag an error if there's a real problem — don't be overly cautious."*

**Tools**: `python_sandbox` (can run its own verification code), `schema_inspector`.

**Model**: Sonnet — this one needs to be good, otherwise it either passes everything or rejects everything.

### 2.5 Writer

**Responsibility**: assembles the results approved by the Critic into a clear, natural-language analysis. Responsible for the style and narrative of the final output.

Concrete tasks:

- Phrase the main answer to the question.
- Highlight numerical findings in text.
- Reference the chart if there is one.
- If the Planner had a multi-step analysis, weave the parts into a coherent narrative.
- Mention key caveats (e.g., low number of data points, NaN distortion).

**Input**: the original question + Planner plan + result from each step + Critic verdicts + Visualizer caption.

**Output**:

```json
{
  "report_text": "During 2023, the West-Transdanubia region...",
  "key_findings": ["West-Transdanubia: +18% growth", "..."],
  "caveats": ["Data covers only two years, so the trend is short-term"]
}
```

**System prompt outline**: *"You write data analysis reports. The input contains the question and the quantitative results. Write a 2-4 paragraph textual analysis: answer to the question, key numbers named, reference to the chart (if any), one sentence on caveats. Don't invent data — only use what you receive. Style: professional but readable."*

**Tools**: none. Pure text generation from the structured input.

**Model**: Sonnet — language quality and factual accuracy are critical here.

### 2.6 Orchestrator (not an agent, but the control logic)

LangGraph-based control logic that:

- Executes the Planner's plan, respecting `depends_on` order
- Handles re-runs when the Critic rejects (max 2 iterations)
- Calls the Router (bandit) before each decision
- Logs every agent call, tool call, latency, and cost into a JSON trace
- Returns the Writer's output + chart + (optionally) the full trace at the end

---

## 3. Tools

6 tools, mostly native Python functions. Where a **stable external MCP server** is available and easier to plug in, we use that. We do not write our own MCP servers.

Unified interface:

```python
def tool(input: dict) -> ToolResult:
    # ToolResult: {success, output, error, latency_ms, cost_estimate}
```

### 3.1 `python_sandbox`

**What it does**: runs Python code in an isolated environment. pandas/numpy/scipy/matplotlib/seaborn available. Time limit (30s), memory limit.

**Implementation — preference order**:

1. **External MCP server**, if we find a reliable one (e.g., `pyodide-mcp` or an `e2b` integration). Provides isolation and sandbox rules for free.
2. Fallback: local `subprocess` with restricted env, or Docker container.

**Never use `exec()`** on untrusted code.

**Input**: `{"code": "...", "timeout": 30}`
**Output**: `{"stdout": "...", "stderr": "...", "return_value": ..., "files_created": [...]}`

### 3.2 `sql_query`

**What it does**: loads the CSV into SQLite and runs an SQL query against it.

**Implementation — preference order**:

1. **External MCP server**: `mcp-server-sqlite` (Anthropic reference server) is exactly for this, directly usable.
2. Fallback: native `pandas.read_csv() → df.to_sql() → sqlite3.execute()`.

**Input**: `{"query": "SELECT region, SUM(revenue) FROM data GROUP BY region"}`
**Output**: `{"rows": [...], "columns": [...]}`

### 3.3 `schema_inspector`

**What it does**: returns metadata about the CSV — columns, types, NaN ratio, number of unique values, sample rows, numeric statistics.

**Implementation**: native Python function. (No meaningful external MCP for this, and it's trivial anyway.)

**Input**: `{"path": "..."}`
**Output**: structured schema dict.

### 3.4 `statistical_test`

**What it does**: runs common statistical tests (t-test, chi-square, ANOVA, correlation). Saves the DataAnalyst from writing scipy code itself — safer.

**Implementation**: native Python function over scipy. (No well-established external MCP for this, and the function is trivial.)

**Input**: `{"test": "t_test", "group1": [...], "group2": [...]}`
**Output**: `{"statistic": ..., "p_value": ..., "interpretation": "..."}`

### 3.5 `chart_generator`

**What it does**: high-level chart API. The Visualizer specifies the type and data, the tool generates a PNG.

**Implementation**: native Python function over matplotlib.

**Input**: `{"chart_type": "bar", "data": {...}, "title": "...", "x_label": "...", "y_label": "..."}`
**Output**: `{"image_path": "..."}`

### 3.6 `file_io`

**What it does**: read/write to the outputs directory. For saving the final report.

**Implementation — preference order**:

1. **External MCP server**: `mcp-server-filesystem` (Anthropic reference) — secure, scopable to specific paths, zero setup.
2. Fallback: native Python `open()`.

**Input**: `{"action": "write", "path": "...", "content": "..."}`
**Output**: `{"success": ...}`

### MCP integration — architecturally

Where an external MCP server is plugged in, the `langchain-mcp-adapters` package automatically converts the MCP tool list into LangChain `Tool` objects. So on the agent side there's no difference — they're called the same way.

**Important principle**: MCP adds complexity (separate processes, async). So **we start with a native version**, measure how it works, and only migrate to MCP where it's simpler. The migration doesn't have to be done for all three tools above — the criterion is: where does it actually save time.

---

## 4. Self-Improving Components

Two components, combined.

### 4.1 Bandit-based router

**What it learns**: at each substep, which (agent, tool) combination to choose.

**Formally**:

- **State (context)**: feature vector from the substep: `[task_embedding (sentence-transformers, 384 dim), question_type (one-hot: agg/filter/stat/viz/compare/text), csv_size_bucket, num_columns_bucket]`. Initially bag-of-keywords is enough instead of the embedding.

- **Action**: discrete set, 8-12 elements, e.g.:
  - `(DataAnalyst, python_sandbox)`
  - `(DataAnalyst, sql_query)`
  - `(DataAnalyst, python_sandbox + statistical_test)`
  - `(Visualizer, chart_generator)`
  - `(Visualizer, python_sandbox)`
  - `(Writer, none)`
  - `(Critic, python_sandbox)` — deep verification
  - `(Critic, schema_inspector_only)` — shallow

- **Reward**: `r = success_score − λ_cost · cost − λ_time · time` where:
  - `success_score`: 1 if Critic approved + final answer is correct, 0 otherwise (or continuous LLM-judge score 0-1)
  - `cost`: total tokens normalized
  - `time`: wall-clock seconds normalized
  - λs are tunable, e.g., `λ_cost=0.1, λ_time=0.05`

**Algorithm**: **LinUCB**, about 50 lines of numpy code.

```
LinUCB initialization:
  For every action a: A_a = I (d×d), b_a = 0 (d)

Action selection given state x:
  For each a: theta_a = A_a⁻¹ · b_a
              p_a = theta_aᵀ · x + α · √(xᵀ · A_a⁻¹ · x)
  Choose: argmax_a p_a

Update (after state x, action a, reward r):
  A_a += x · xᵀ
  b_a += r · x
```

**Where it fits in**: the Planner *suggests* in the plan, but the Router can override at each step. Both are logged — in the report we can compare how often the router deviated from the Planner's suggestion.

**Learning process**:

1. Cold start: starts from the Planner's decisions with small exploration noise.
2. After each completed workflow, (state, action, reward) triplets go into a buffer.
3. Online update after every new example — LinUCB is incremental, no batch retraining needed.

### 4.2 Skill / error memory

**What it learns**: stores and retrieves concrete, reusable patterns.

**Two memories**:

1. **Skill library** — successful code snippets. E.g., "monthly aggregation from a date column" → stored code template. For similar future tasks, vector search retrieves the top-3 examples as few-shot in the system prompt.
2. **Error catalog** — errors found by the Critic. (bad_code, error_type, fix) triplets. For similar tasks, the DataAnalyst gets a proactive warning: *"watch out, NaN issues are common in tasks like this."*

**Implementation**: ChromaDB (or FAISS), with embedded task descriptions as keys. Sentence-transformers `all-MiniLM-L6-v2` for embedding.

**When it reads**: before every DataAnalyst call — top-3 skills + top-2 errors retrieved.

**When it writes**:

- Skill: at the end of the workflow, if success_score > 0.8 — stores the (task, code) pair.
- Error: when the Critic says `needs_revision` — stores the (task, bad_code, error_description, fix) tuple.

**Why this is more than "just" RAG**: the system **populates the database itself** after each run. This is what makes it self-improving.

### 4.3 How the two components work together

A single run looks like:

1. User question arrives
2. `schema_inspector` runs on the CSV → schema metadata
3. **Planner** decomposes into substeps (creates a plan)
4. For every substep:
   - **Router (bandit)** chooses the (agent, tool) pair based on the state
   - The agent looks at the **skill library** for similar prior successes + the **error catalog** for typical mistakes → these go into the system prompt
   - The agent executes the task
   - **Critic** verifies; on error → goes into the error catalog + possibly re-run
5. **Writer** assembles the final textual report
6. End of workflow: success_score is computed, rewards are propagated to each router decision, **bandit updates**, **skill library grows**

For the report's argument: this setup combines three RL-flavored mechanisms — exploration/exploitation in the bandit, experience replay in the skill memory, and credit assignment in the reward structure. More than enough to defend the "reinforcement learning" framing.

---

## 5. Performance Measurement

### 5.1 Test dataset

**30-50 items, manually curated**, with this schema:

```json
{
  "id": "task_001",
  "csv_path": "datasets/sales_2023.csv",
  "question": "Which region had the largest revenue growth in 2023?",
  "difficulty": "medium",
  "category": "aggregation_comparison",
  "expected_answer": {
    "type": "categorical",
    "value": "West-Transdanubia",
    "tolerance": null
  },
  "expected_chart": true,
  "expected_text_keywords": ["growth", "West-Transdanubia", "region"],
  "reference_code": "df.groupby(['region', 'year'])..."
}
```

**Categories**: `simple_filter`, `aggregation`, `group_compare`, `time_series`, `statistical_test`, `multi_step`, `with_visualization`, `with_text_analysis`. 4-6 items per category.

**Data sources**: Kaggle (Titanic, retail, sales), self-generated CSVs for controlled cases, or a subset of **InfiAgent-DABench**.

The dataset is split into **train/test**: 60% train (the bandit and memory learn on this), 40% holdout test (only for the final measurement).

### 5.2 Metrics

**1. Correctness metrics (the main indicator)**

- **Exact match rate**: ±1% tolerance for numeric, string match for categorical.
- **LLM-as-judge score (0-5)**: a strong model (Sonnet/Opus) evaluates the final report given the `expected_answer` — this is the main metric for the Writer's output.
- **Code correctness**: did the generated code run (binary).
- **Chart correctness**: if a chart was needed, was it generated and is the content (labels, datapoints) plausible (LLM-judge).
- **Text keyword recall**: does the Writer's output contain `expected_text_keywords`.

**2. Cost metrics**

- Total input + output tokens, broken down by agent
- Tool call counts by type
- Estimated API cost in dollars
- Wall-clock time

**3. Workflow metrics**

- Trajectory length (how many steps)
- Number of Critic re-runs
- Tool error rate

**4. Learning metrics (only for the self-improving version)**

- Reward over time curve
- Bandit policy entropy (how decisive the router was)
- Skill library hit rate (how often it found a relevant prior skill)
- Cumulative regret (chosen action vs. retrospectively best)

### 5.3 Measurement system architecture

A self-written Python package, three modules:

- **`runner.py`**: iterates over the test dataset, calls the multi-agent system on each task, saves a JSON trace of every agent and tool call.
- **`evaluator.py`**: runs the metrics on the traces. Contains the exact-match logic, the LLM-judge, and the learning metrics.
- **`reporter.py`**: aggregated report — success rate per category, cost distribution, ablation table, learning curves with matplotlib.

The structured trace logging **serves two purposes simultaneously**: measurement and training of the self-improving component.

### 5.4 Ablation study

The most convincing part of the report will be this table:

| Setup | Success rate | Avg cost | Avg time |
|---|---|---|---|
| Single-agent baseline (DataAnalyst only) | ? | ? | ? |
| Multi-agent, static routing (Planner decides) | ? | ? | ? |
| Multi-agent + skill memory | ? | ? | ? |
| Multi-agent + bandit router | ? | ? | ? |
| Full system (router + memory) | ? | ? | ? |

### 5.5 Statistical significance

We run each setup with 3-5 different seeds (LLM temperature seed + bandit init seed). Compute mean + std, and bootstrap confidence intervals on the 30-50 item holdout.

---

## 6. Implementation Phases

Week-by-week breakdown according to the work plan.

### Weeks 1-4: Literature review

Concrete topics to read:

- **Multi-agent LLM architectures**: AutoGen, MetaGPT, ChatDev papers; LangGraph documentation.
- **Tool use and function calling**: ReAct paper, Toolformer, OpenAI/Anthropic function calling guides.
- **MCP (Model Context Protocol)**: spec, reference servers, `langchain-mcp-adapters`.
- **RL fundamentals**: Sutton-Barto Ch. 2 (multi-armed bandits), Ch. 17 (contextual bandits). Original LinUCB paper (Li et al. 2010).
- **Agentic RL**: Voyager paper (Wang et al. 2023), Reflexion (Shinn et al. 2023), Generative Agents (Park et al. 2023).
- **Data analysis benchmarks**: InfiAgent-DABench paper, DABStep, Spider 2.0.

By end of week 4: 8-10 page summary of the key papers, motivation for the architectural decisions.

### Week 5: Skeleton + base architecture

**Goal**: end-to-end working "skeleton" with **1 agent, 1 tool**, so that the pipeline stands.

Concrete tasks:

1. Repo init, dependencies (`langgraph`, `langchain`, `chromadb`, `sentence-transformers`, `pandas`, `matplotlib`, `langchain-mcp-adapters`).
2. `python_sandbox` tool implementation (subprocess-based or MCP).
3. Simple DataAnalyst agent that takes a question + CSV, generates code, runs it, returns the result.
4. **Eval pipeline minimum version**: 5 manually written test cases, runs → success rate. Don't drop this until the end — it runs after every change.
5. Logging structure: every LLM call and tool call structured into JSON.

**By end of week**: working end-to-end demo, runs on 5 examples.

### Week 6: Full agent architecture + all tools + metrics

**Goal**: the vanilla (non-self-improving) system is done.

Tasks:

1. Implementation of **all 5 agents** (Planner, DataAnalyst, Visualizer, Critic, Writer).
2. Implementation of **all 6 tools** (native Python; where available and easier, MCP).
3. LangGraph orchestrator: state schema, nodes, edges, dependency handling, re-run logic.
4. **Performance measurement**: creation of the 30-50 item test dataset. First version of `runner.py`, `evaluator.py`, `reporter.py`.
5. **Baseline measurement**: run the full dataset on the vanilla system. This is the baseline number in the ablation table.

**By end of week**: vanilla system done + baseline numbers in.

### Week 7: Skill / error memory

**Goal**: the first self-improving component — the memory.

Tasks:

1. ChromaDB setup, embedding pipeline (sentence-transformers).
2. Skill library: write logic at end of workflow, retrieval before DataAnalyst calls.
3. Error catalog: write at Critic re-run trigger, retrieval into DataAnalyst.
4. Few-shot integration into system prompts (top-3 skills + top-2 errors into context).
5. **Measurement**: re-run the full dataset, comparison with vanilla baseline.

**By end of week**: memory version done, measured improvement.

### Week 8: Bandit router design + integration

**Goal**: the second self-improving component — learned routing.

Tasks:

1. State feature extractor: bag-of-keywords + optionally sentence-transformer embedding.
2. Precise definition of the action space (8-12 (agent, tool) combinations).
3. **LinUCB implementation** in numpy, ~50 lines.
4. Reward function implementation: success_score (Critic + LLM-judge), cost normalization, time normalization.
5. Integration into the orchestrator: Planner suggests, bandit decides or overrides.
6. Logging: every router decision (state, suggested_action, chosen_action, reward) stored.

**By end of week**: bandit router works, but not yet trained — cold start mode.

### Week 9: Bandit training + tuning + ablation

**Goal**: the full self-improving system comes together, measured per-category results.

Tasks:

1. **Bandit cold start training**: on 60% of the train set (~18-30 examples), run multiple times (5-10 epochs), the bandit learns.
2. λ-hyperparam tuning: measure with different `λ_cost`, `λ_time` values to see the tradeoff.
3. Verify the interaction between skill library and bandit — they don't learn against each other.
4. **Full ablation run** for all 5 setups, with 3 seeds:
   - Single-agent baseline
   - Multi-agent static
   - Multi-agent + memory
   - Multi-agent + bandit
   - Full system
5. Plotting learning curves.

**By end of week**: all numbers in for the report.

### Week 10: Testing and analysis

Tasks:

1. **Holdout test run** — the 40% holdout dataset that the components haven't seen yet.
2. Statistical significance: bootstrap CI on the main metrics.
3. **Error analysis**: in which categories did the self-improving component help, in which did it not? Concrete examples.
4. Reproducibility check: does the same seed → same result?
5. Visualizations for the report: learning curves, ablation bar chart, per-category comparison.

**By end of week**: every figure, table, number is done.

### Weeks 11-12: Report

**Suggested report structure**:

1. **Introduction** (1-2 pages): problem, motivation, research questions.
2. **Literature review** (3-4 pages): multi-agent architectures, RL in agentic context, tool usage.
3. **System architecture** (3-4 pages): agents, tools, orchestrator, justification of architectural decisions.
4. **Self-improving component** (3-4 pages): bandit router formally, skill memory, integration.
5. **Performance measurement** (2-3 pages): test dataset, metrics, measurement system architecture.
6. **Results** (3-4 pages): ablation table, learning curves, per-category analysis, statistical significance, error analysis.
7. **Discussion** (1-2 pages): what the system learned, where it didn't help, limitations.
8. **Conclusion and future directions** (1 page).

Week 11: first version to consultant, by end of the week.
Week 12: revisions, final version.

---

## 7. Summary checklist

- ✅ **5 agents**: Planner, DataAnalyst, Visualizer, Critic, Writer
- ✅ **6 tools, varied in nature**: code execution, querying, metadata, statistics, visualization, I/O
- ✅ **Tools are primarily native Python functions**, where a stable external MCP server is available and easier, we use it
- ✅ **No custom MCP servers written**
- ✅ **Self-improvement with bandit** (LinUCB) + skill/error memory
- ✅ **Self-built measurement system**: 30-50 item curated dataset, train/test split, 5-row ablation, statistical significance with 3-5 seeds
- ✅ **Work-plan-aligned** schedule for the 12-week semester
