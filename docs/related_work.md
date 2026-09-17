# Related work and reading map

This list is intentionally broader than the paper's final bibliography. A work
is included when even one part bears on the project: predicting correctness,
probe methodology, causal interpretation, activation intervention, confidence,
or evaluation design. Links point to arXiv, ACL Anthology, OpenReview, or the
authors' technical publication rather than to search summaries.

## Closest work: predicting correctness from internal states

1. [No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](https://arxiv.org/abs/2509.10625) (Cencerrado et al., 2025/2026). The closest match to the prompt-before-generation probe. It finds question-only correctness directions across larger models, with weaker transfer to mathematical reasoning. Question-only correctness probing therefore precedes this study.

2. [Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](https://arxiv.org/abs/2605.09502) (Yuan et al., 2026). The closest match to the broad “strong probe, failed intervention” result. It probes generated reasoning states, while this project probes the prompt-final state and emphasizes matched random-direction and random-subspace controls. The current SDQ results do not establish that correctness information is non-causal.

3. [Probing for Arithmetic Errors in Language Models](https://aclanthology.org/2025.emnlp-main.411/) (Sun, Stolfo, and Sachan, 2025). Trains hidden-state error detectors in controlled addition and transfers them to addition-focused GSM8K traces. Directly relevant to the arithmetic family and to the possibility of using a detector for selective re-prompting.

4. [Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](https://aclanthology.org/2026.acl-long.483/) (Ashuach et al., 2026). Compares probes on a model's own representations with probes on peer-model representations. The lack of a privileged self-signal in math is a useful warning: a correctness probe may largely read shared problem difficulty.

5. [Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/abs/2406.15927) (Kossen et al., 2024). Predicts semantic entropy from one generation's hidden states. Relevant because “error signal” may actually be a representation of uncertainty rather than a causal correctness variable.

6. [Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation](https://aclanthology.org/2025.findings-emnlp.880/) (2025). Shows that lightweight hidden-state probes can predict long-form factuality and tests out-of-distribution transfer. Useful as a broader error-detection precedent beyond short synthetic reasoning.

7. [ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs](https://aclanthology.org/2025.acl-long.880/) (Zhang et al., 2025). Uses cross-layer residual updates rather than one static state. This relates to the historical trajectory emphasis and the distinction between dynamic features and a single-state readout.

8. [Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs](https://aclanthology.org/2025.emnlp-main.238/) (Tan et al., 2025). Shows that common uncertainty and probe-based detectors struggle on errors repeated across samples. Relevant to the limitation of greedy, one-sample labels.

9. [ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://aclanthology.org/2025.acl-long.50/) (Zheng et al., 2025). Provides human-annotated erroneous steps in mathematical reasoning. It distinguishes process-error localization from the sequence-level correctness target used here.

## Self-evaluation, latent knowledge, and confidence

10. [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) (Kadavath et al., 2022). Studies verbalized probabilities of answer correctness and “I know” predictions. It motivates a behavioral comparator and also shows why correctness prediction should not automatically be called hidden awareness.

11. [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827) (Burns et al., 2022). Finds unsupervised truth-related directions from logical consistency. Relevant as a contrast to the supervised outcome probe and to the question of whether the direction is dataset-specific.

12. [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824) (Marks and Tegmark, 2023/2024). Combines transfer, difference-of-means directions, and causal interventions. It is the strongest positive contrast: some simple truth directions do have a behavioral effect.

13. [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112) (Belrose et al., 2023). Decodes evolving token predictions at intermediate layers and performs causal checks. Relevant to interpreting layer trends without pretending raw residual states are already in output-logit coordinates.

14. [Your Reasoning Model is Secretly a Reward Model: Optimization-Free Verification from Experience](https://aclanthology.org/2026.acl-long.788/) (2026). Uses hidden-state trajectories for binary correctness verification in tasks with deterministic outcomes. Useful context for the broader family of internal-verifier work.

## How to interpret probes

15. [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/) (Hewitt and Liang, 2019). Introduces control tasks and selectivity. It supports using explicit non-neural and nuisance baselines rather than treating raw probe accuracy as self-interpreting.

16. [Probing Classifiers: Promises, Shortcomings, and Advances](https://aclanthology.org/2022.cl-1.7/) (Belinkov, 2022). A compact review of probe capacity, baselines, metrics, and causal limitations. This is the best single methodological overview to cite.

17. [Information-Theoretic Probing for Linguistic Structure](https://aclanthology.org/2020.acl-main.420/) (Pimentel et al., 2020). Frames probing as estimating information. Relevant to the limited claim “the state contains extractable predictive information.”

18. [Information-Theoretic Probing with Minimum Description Length](https://aclanthology.org/2020.emnlp-main.14/) (Voita and Titov, 2020). Measures how easily labels can be extracted, rather than only final accuracy. This is relevant to the sample-efficiency and probe-capacity limitations of AUROC-only evaluation.

19. [Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?](https://aclanthology.org/2021.eacl-main.295/) (Ravichander, Belinkov, and Hovy, 2021). Demonstrates that a model can encode properties it does not need for its task. This directly undercuts any jump from decodability to use.

20. [Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals](https://aclanthology.org/2021.tacl-1.10/) (Elazar et al., 2021). Tests use by removing linearly decodable information and comparing with random subspace removal. Its matched-random control logic is particularly close to this project's strongest design choice.

21. [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://aclanthology.org/2020.acl-main.647/) (Ravfogel et al., 2020). Introduces iterative nullspace projection. Relevant if a future robustness check removes the whole linearly decodable subspace instead of adding one probe vector.

22. [Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE](https://aclanthology.org/2025.findings-acl.674/) (Dobrzeniecka, Fokkens, and Sommerauer, 2025). Shows that information-removal methods themselves can inject nonspecific changes. This reinforces the need for geometry-matched controls and output-quality reporting.

## Activation steering and causal intervention

23. [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://openreview.net/forum?id=aLLuYpn83y) (Li et al., NeurIPS 2023). Selects attention heads with probes and shifts activations along their truthful directions. This is a central positive precedent for treating a probe weight as a candidate steering vector.

24. [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248) (Turner et al., 2023). Introduces Activation Addition using contrastive activation differences. Relevant to the basic additive intervention and to measuring off-target damage.

25. [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (Panickssery et al., 2023/2024). Develops and evaluates contrastive activation addition across behavioral traits. A useful reference for layer, position, and coefficient choices.

26. [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) (Zou et al., 2023). Places population-level representation reading and control in a general framework. The present paper is best read as a boundary case for this program, not a refutation of it.

27. [Activation Scaling for Steering and Interpreting Language Models](https://arxiv.org/abs/2410.04962) (Stoehr et al., 2024). Separates intervention effectiveness, off-target faithfulness, and sparsity. Those distinctions are useful for categorizing answer changes versus degeneration.

28. [Steering Large Language Models with Feature Guided Activation Additions](https://arxiv.org/abs/2501.09929) (Soo, Teng, and Balaganesh, 2025). Evaluates steering on Gemma 2 2B and 9B using SAE-guided features. Same-model evidence that the architecture can be steered does not by itself validate this project's particular hook and direction, but it helps motivate a local positive control.

## Causal localization and experimental controls

29. [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (Meng et al., 2022). Causal tracing localizes components that matter to factual recall and ROME separately edits weights. Relevant as a careful example of moving from correlation to intervention.

30. [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042) (Zhang and Nanda, 2023/2024). Shows that patching conclusions depend strongly on the metric and corruption method. It supports predeclaring the intervention surface and using a positive control.

31. [Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models](https://arxiv.org/abs/2301.04213) (Hase et al., 2023). Finds that where a causal-localization method points need not be the best place to edit. This is an especially useful analogy for why a predictive direction need not be an effective control direction.

32. [Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) (Chan et al., 2022, Redwood Research technical article). Formalizes interpretability hypotheses through behavior-preserving resampling ablations. Relevant to testing whether a hypothesized computation accounts for behavior, rather than only predicting labels.

## Relationship to the present study

Correctness probes, prompt-before-generation prediction, and failed steering
each have prior work. The present study combines:

1. controlled paraphrase groups held out by underlying semantic problem;
2. comparison against prompt-text and next-token-confidence baselines;
3. norm- and subspace-matched controls for additive intervention;
4. prompt-clustered inference across repeated random directions; and
5. transparent separation of an exploratory effect from its failed confirmation.
