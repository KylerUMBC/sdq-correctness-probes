# Certification Protocol

SDQ must be certified with **structural tests**, not just retrieval/similarity metrics.

## Required Tests

### A. Alignment Validity
Do semantically equivalent trajectories align under monotone time warps better than random controls?

### B. Transport Validity
Do local transports explain equivalent-prompt differences better than shuffle controls?

### C. Cycle Validity
Do learned transports approximately invert ($G^{j \to i} G^{i \to j} \approx I$)
and compose ($G^{j \to k} G^{i \to j} \approx G^{i \to k}$)?

### D. Transfer Validity
Do transports learned on one semantic family transfer to another?

### E. Same-Answer Control
Can the model distinguish same-answer / different-reasoning cases?

### F. Hard-Negative Control
Can it resist collapsing surface-near but semantically-different prompts?

### G. Event Preservation
Do aligned trajectories preserve reasoning-stage ordering?
