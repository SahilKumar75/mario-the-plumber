# Mario vs Reference OpenEnv Repos

This note compares Mario against the official OpenEnv reference repos mentioned in the hackathon material.

## Calendar Env

What Calendar shows:

- OpenEnv can wrap a real service surface, not just a simulator

Where Mario differs:

- Mario is more benchmark-centric and task/grader-oriented
- Mario models ETL repair and recovery explicitly instead of generic tool use

## Reasoning Gym Env

What Reasoning Gym shows:

- a very small and elegant benchmark can still be valid

Where Mario differs:

- Mario is heavier because it models multi-step operational recovery
- Mario gains realism and richer reward structure at the cost of minimalism

## TBench2 Env

What TBench2 shows:

- infrastructure depth and execution backends matter for operational benchmarks

Where Mario differs:

- Mario is less backend-diverse
- Mario is easier to run and score deterministically
- Mario focuses on ETL recovery instead of terminal task execution

## CARLA Env

What CARLA shows:

- serious scenario presentation and a strong visualization story increase reviewer confidence

Where Mario differs:

- Mario is not simulator-heavy or GPU-heavy
- Mario compensates with benchmark clarity, deterministic graders, and business-domain realism

## REPL Env

What REPL shows:

- complex benchmarks benefit from cleaner separation between orchestration, execution, and reward logic

Where Mario differs:

- Mario still has some heavier core files
- the internal benchmark package now closes part of that gap by separating catalog, grading, and policy logic

## Summary

Mario is strongest as a benchmark-specific ETL recovery environment:

- stronger benchmark framing than Calendar
- richer real-world ops structure than Reasoning Gym
- lighter and easier to validate than CARLA
- more domain-specific than TBench2 and REPL

The main design goal is not to imitate every reference repo equally. It is to stay focused on broken ELT/ETL pipeline recovery while reaching comparable benchmark seriousness.
