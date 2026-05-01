# qmlkit Adoption Plan

This plan is for getting qmlkit from a polished public alpha to real users.

## Positioning

Lead with:

> Quantum-inspired ML that works in sklearn and PyTorch.

Avoid leading with:

- quantum SDK;
- quantum advantage;
- hardware execution;
- circuit framework.

The first audience is ML engineers who are curious about quantum-inspired
methods but do not want physics overhead.

## Success Metrics

Track these weekly:

- GitHub stars, forks, and watchers;
- unique cloners and visitors from GitHub traffic;
- issues opened by people other than the maintainer;
- examples run successfully by first users;
- external mentions on Hacker News, Reddit, X, LinkedIn, and newsletters.

## First 100 Users

Goal: get feedback, not scale.

Actions:

- Ask 10-20 ML engineers or students to run one example.
- Ask each tester where the README became unclear.
- Open issues for every repeated confusion.
- Publish one short post showing `QuantumKernel + SVC`.
- Add one honest benchmark where qmlkit helps and one where it does not.

## First 500 Users

Goal: make the repo shareable.

Actions:

- Add three notebooks:
  - quantum-inspired SVM on small nonlinear data;
  - drop-in PyTorch `HybridLayer`;
  - honest benchmark failures and limitations.
- Submit a restrained Show HN post.
- Post a technical write-up to Reddit and LinkedIn.
- Add a gallery table to the README with datasets, baselines, and results.

## First Contributors

Goal: make contribution obvious.

Good first issues:

- add a dataset benchmark;
- add an example notebook;
- improve sklearn pipeline ergonomics;
- add Qiskit/PennyLane adapter tests for more circuit shapes;
- improve docs around when feature maps help.

Maintainer response target:

- respond to first-time contributors within 48 hours;
- accept small docs/example improvements quickly;
- keep roadmap issues scoped to one pull request.

## Messaging

Recommended launch title:

> Show HN: Quantum-inspired ML that runs on CPU and plugs into sklearn

Short social post:

> I open-sourced qmlkit, a small Python SDK for trying quantum-inspired ML
> without learning circuit DSLs first. It gives ML engineers sklearn-style
> feature maps, a PyTorch hybrid layer, local Qiskit/PennyLane adapters, and
> honest toy benchmarks. Public alpha: feedback welcome.

## Product Roadmap

Near term:

- benchmark against RBF SVM, random forests, gradient boosting, and MLPs;
- add notebooks with clear failure cases;
- add examples using real tabular datasets;
- add visualizations for feature maps and decision boundaries.

Later:

- richer `SVC(kernel="precomputed")` support;
- PyTorch modules with clearer initialization controls;
- Qiskit/PennyLane adapter expansion;
- docs site once APIs stabilize.
