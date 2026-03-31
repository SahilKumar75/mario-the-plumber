# Mario the Plumber: RL-Grounded Benchmark Review

This note updates the project assessment using ideas from the following materials:

- Ben Hambly et al., *Recent advances in reinforcement learning in finance* (2023)
- Jesse Clifton and Eric Laber, *Q-Learning: Theory and Applications* (2020)
- Yuxi Li, *Deep Reinforcement Learning: An Overview* (2018)
- Ashish Kumar Shakya et al., *Reinforcement learning algorithms: A brief survey* (2023)
- Cédric Vandelaer, *Reinforcement Learning: An introduction (Part 1/4)*
- Niels Justesen et al., *Illuminating Generalization in Deep Reinforcement Learning through Procedural Level Generation* (2018)
- Alberto Maria Metelli, *Configurable Environments in Reinforcement Learning: An Overview* (2022)
- Lei Yuan et al., *A Survey of Progress on Cooperative Multi-Agent Reinforcement Learning in Open Environment* (2023)
- Rui Wang et al., *Enhanced POET: Open-ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions* (2020)

The goal is not to restate RL theory. The goal is to judge whether Mario the Plumber behaves like a meaningful RL / agent benchmark and to identify the highest-value next improvements.

## Additional Takeaways from the Second Reading Pass

The newer papers sharpened four ideas that matter directly for Mario:

### Generalization matters more than single-seed success

Justesen et al. show that RL systems often overfit to fixed environments and can look stronger than they really are when episode diversity is too low. That strengthens the case for Mario's move toward seed variation and harder Task 3 distributions. It also means Mario should keep growing the variety of corruption patterns instead of optimizing too hard for one canonical seed.

### Configurable environments are a benchmark strength, not a side feature

Metelli's overview of configurable RL environments supports an important framing choice: exposing task IDs, seeds, and controllable scenario generation is a feature of a serious benchmark. It makes evaluation more systematic, lets us stress-test generalization, and supports curriculum-like analysis without changing the agent interface.

### Open environments reward robustness, not just optimal trajectories

The open-environment MARL survey emphasizes that real environments change, expand, and surprise agents. For Mario, the single-agent version of that lesson is: the benchmark gets stronger when table states, error mixes, and dependency structures vary enough that agents must adapt instead of replaying one memorized cleanup script.

### Open-ended challenge generation is a long-term direction

Enhanced POET reinforces the value of environments that keep producing novel but still learnable challenges. Mario is not open-ended today, but the paper is a strong argument for future work where corruption generators evolve or diversify automatically instead of being hand-authored forever.

## 1. Research-Grounded Rating

Current honest score: **89/100**

Breakdown:

- Real-world utility: **26/30**
- Task and grader quality: **22/25**
- Environment design: **18/20**
- Code quality and spec compliance: **14/15**
- Creativity and novelty: **9/10**

Why this score is now defensible:

- The environment is live, valid, and reproducible.
- Task 3 now has a real benchmark gap:
  - initial score over 20 seeds: average `0.2005`
  - random agent over 20 seeds: average `0.2065`
  - structured baseline on seed `42`: `0.9070`
- The benchmark therefore distinguishes random behavior from structured repair.
- The task domain is realistic enough to matter: table repair, schema restoration, duplicate handling, and cross-table repair sequencing are all real operational problems.

Why it is not yet a 95+ benchmark:

- The provided baseline is still hybrid heuristic + LLM rather than a pure LLM or learned RL policy.
- Task 3 is materially harder than before, but still not proven hard for frontier evaluators such as Nemotron-class agents.
- Corruption types are still narrower than the full range of real production data failures.

## 2. Five Highest-Value Improvements

These are ordered by practical impact rather than academic elegance.

### Improvement 1: Measure a pure-LLM baseline on Task 3

Why:

- The current benchmark now separates random from structured behavior.
- The missing number is the score of a pure LLM policy without heuristic stabilization.
- That number is the clearest evidence for whether Task 3 really needs model reasoning.

What to do:

- Add a switch in `inference.py` to disable heuristic overrides.
- Benchmark Task 3 on 5 to 10 seeds with at least 2 different models.
- Report:
  - random score
  - pure LLM score
  - hybrid score

Target outcome:

- random `~0.20`
- pure LLM meaningfully above random
- hybrid still highest

That 3-level ranking would make the benchmark story much stronger.

### Improvement 2: Expand corruption realism, not just difficulty

Why:

- The reviewed RL literature repeatedly emphasizes that good environments should reflect the true structure of the decision problem, not only be hard.
- Mario currently models nulls, duplicates, outliers, type drift, and derived-field inconsistency well.
- It still under-represents real data engineering failures such as inconsistent dates, encoding noise, and schema drift across systems.

High-ROI additions:

- mixed date formats in a date column
- text encoding corruption in a categorical/string column
- unit mismatch such as price in cents versus dollars
- upstream enum/category drift

These make the benchmark more believable to humans and more semantically challenging to agents.

### Improvement 3: Separate benchmark evaluation baselines from training baselines

Why:

- Hambly et al. and the Q-learning review both reinforce the point that reward signals and behavior policies serve different roles depending on whether the goal is evaluation or learning.
- Mario is currently excellent as an evaluation environment.
- It is less mature as a training environment because some reward behavior is intentionally shaped around safe repair heuristics.

What to do:

- Keep the current baseline as the official submission baseline.
- Add a second documented baseline:
  - `safe_eval_baseline`
  - `pure_policy_baseline`

This makes the benchmark easier to discuss honestly with reviewers.

### Improvement 4: Improve observation semantics for multi-table reasoning

Why:

- The deep RL overview highlights memory, hierarchy, and planning as the mechanisms that make more complex decision problems meaningful.
- Mario Task 3 currently has cross-table dependency, but the observation is still mostly a flat error summary.

What to add:

- explicit upstream/downstream dependency hints in the observation
- table-level health summaries side by side
- a “commit risk” hint when derived fields depend on still-broken upstream tables

This would increase the planning aspect without making the action space larger.

### Improvement 5: Add benchmark-style reporting artifacts

Why:

- Research benchmarks become stronger when they report comparisons, not just a single successful run.

What to publish:

- seed-sweep table
- random baseline table
- hybrid baseline table
- optional pure-LLM baseline table
- task-by-task difficulty notes

This turns the project from “working submission” into “small but credible benchmark.”

## 3. A Stronger Task 4 Design

If Mario gets extended, the best next task is not “more of the same corruption.” It should force longer-horizon reasoning and better match real data operations.

### Proposed Task 4: Incremental Pipeline Recovery

Scenario:

- Three upstream tables arrive in batches.
- One batch is partially loaded.
- A downstream summary table has already been computed from stale or incompatible inputs.
- The agent must detect whether to repair data, reconcile schema drift, or avoid committing too early.

Core idea:

- The agent should no longer be able to win by simply cleaning one active snapshot.
- It must reason about:
  - batch freshness
  - cross-table consistency
  - derived-table invalidation
  - commit timing

Suggested corruption types:

- partial load in `orders`
- schema drift in `customers.signup_date`
- price-unit mismatch in `products`
- stale aggregate table with wrong totals
- duplicate or delayed late-arriving rows

Suggested action extensions:

- validate batch completeness
- recompute downstream summary
- rollback latest transform
- refresh derived table

Why this would be strong:

- It adds sequential dependency, not just more columns with errors.
- It better matches how real ETL failures happen in production.
- It creates a benchmark where order of actions matters more deeply.

## 4. Comparison to a More Serious RL Benchmark

Mario is now a credible benchmark, but it still differs from a top-tier RL benchmark in important ways.

### Where Mario is already strong

- clear episodic structure
- deterministic grading
- meaningful benchmark gap between random and structured behavior
- low operational complexity for evaluators
- real-world task framing rather than games

### Where a stronger benchmark would go further

- broader corruption families
- less heuristic-solvable repair trajectories
- more partial observability
- more explicit long-horizon planning cost
- multiple strong baseline families:
  - random
  - heuristic
  - pure LLM
  - trained RL policy

### Benchmark maturity comparison

Current Mario:

- strong hackathon benchmark
- credible evaluation environment
- partially RL-like, strongly agentic

Stronger research-grade benchmark:

- larger state diversity
- more irreversible action tradeoffs
- richer generalization tests across seeds and failure modes
- clearer separation of evaluation and training settings

### Best summary

Mario is no longer a toy benchmark. After Task 3 hardening, it has the one thing weak benchmarks usually lack: a meaningful gap between random and structured decision-making.

What keeps it below a research-grade benchmark is not correctness. It is ambition:

- the failure space is still relatively compact
- the strongest baseline is still hybrid
- the hardest task is not yet proven difficult for a frontier evaluator

That is a very good place to be for a first OpenEnv environment.

## Practical Takeaway

If the project ends here, it is a strong submission.

If the project continues after the hackathon, the best path is:

1. benchmark a pure LLM baseline
2. expand corruption realism
3. build Task 4 around incremental pipeline recovery
4. publish comparative evaluation tables

That roadmap would move Mario from a strong hackathon environment toward a genuinely reusable agent benchmark.
