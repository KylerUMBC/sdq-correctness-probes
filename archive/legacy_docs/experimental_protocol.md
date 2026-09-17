# Experimental Protocol

## Phases

### Phase 1 — Formalism
- Formalize trajectory space, local transport, time alignment, equivalence relation
- Deliverable: clean math note + markdown spec

### Phase 2 — Pairwise Alignment + Local Transport
- Implement pairwise trajectory alignment (soft-DTW or monotonic attention)
- Learn local transport fields between aligned trajectory pairs
- Test cycle consistency and triple composition
- Deliverable: pairwise SDQ transport prototype

### Phase 3 — Latent Semantic Trajectory Model
- Add latent semantic state encoder
- Add reconstruction decoder
- Semantic alignment loss + joint optimization
- Deliverable: SDQ-v1 end-to-end estimator

### Phase 4 — Event Structure
- Reasoning-stage inference or labels
- Event-preservation constraints
- Stage-aware evaluation
- Deliverable: event-aware SDQ system

### Phase 5 — Strong Certification
- Unseen-template evaluation
- Hard-negative evaluation
- Same-answer / different-reasoning evaluation
- Cycle and transfer tests
- Deliverable: robust structural validation

## Data Layers

1. **Minimal paraphrase families** — synonym substitution, active/passive, clause reordering
2. **Cross-template equivalence** — same reasoning task under different scaffolds
3. **Reasoning-preserving perturbations** — same inference graph, varied surface
4. **Hard negatives** — similar wording, different semantics
5. **Same answer, different reasoning** — same output, different paths
