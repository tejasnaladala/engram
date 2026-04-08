---
name: New Learning Rule
about: Propose or implement a new learning rule
title: "[LearningRule] "
labels: learning-rule, enhancement
---

**Learning rule name**
e.g., BCM, Oja's rule, e-prop, surrogate gradient

**Mathematical formulation**
The weight update equation:

```
dw = ...
```

**Reference**
Paper or textbook reference.

**Why is this useful for Engram?**
What kind of tasks or behaviors does this enable?

**Implementation plan**
- [ ] Implement `LearningRule` trait in `crates/engram-core/src/learning_rule.rs`
- [ ] Add unit tests
- [ ] Add example demonstrating the rule
- [ ] Update docs
