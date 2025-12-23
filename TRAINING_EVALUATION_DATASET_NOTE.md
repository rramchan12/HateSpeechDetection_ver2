# Training and Evaluation Dataset

## Dataset Composition and Unification

The training and evaluation dataset employed in this study comprises a unified corpus of 5,151 samples derived from two complementary hate speech detection datasets: HateXplain (Mathew et al., 2021) and ToxiGen (Hartvigsen et al., 2022). The unification strategy addresses the critical need to capture diverse hate speech modalities while maintaining balanced representation across source distributions, target demographics, and classification labels.

### Source Datasets

**HateXplain** contributes 2,427 samples (47.1% of unified corpus), providing persona-based hate speech with explicit targeting patterns. This dataset contains human-annotated rationales at the token level, offering interpretability signals essential for explainable AI research. The samples exhibit clear attribution of hateful content to specific protected characteristics, with annotators identifying offensive text spans and their relationship to targeted demographic groups.

**ToxiGen** contributes 2,724 samples (52.9% of unified corpus), providing implicit hate speech characterized by subtle toxicity and coded language. Generated using GPT-3 with controlled prompting strategies (96.1% top-k sampling, 3.9% ALICE method), these samples represent context-dependent hate that may evade explicit pattern detection, thereby complementing HateXplain's explicit targeting modality.

### Target Group Taxonomy

The unified dataset focuses on three protected demographic groups representing the most frequent targets in both source corpora:

1. **LGBTQ+** (2,516 samples, 48.8%): Sexual and gender minorities, encompassing homosexual, gay, and queer identities
2. **Middle Eastern** (1,470 samples, 28.5%): Arab and Middle Eastern populations, including Muslim religious identity
3. **Mexican/Latino** (1,165 samples, 22.6%): Hispanic and Latino demographics

This three-group taxonomy was established through persona normalization procedures that mapped HateXplain's target categories (`Homosexual`, `Gay`, `Arab`, `Hispanic`, `Latino`) to ToxiGen's group labels (`lgbtq`, `middle_east`, `mexican`) using a standardized mapping schema. All samples targeting demographic groups outside this taxonomy were excluded to maintain focused representation and evaluation consistency.

### Label Distribution and Balancing

The dataset implements near-optimal binary classification balance achieved through strategic undersampling of the minority source (HateXplain):

- **Hate speech**: 2,426 samples (47.1%)
- **Normal content**: 2,725 samples (52.9%)

This 47:53 distribution approximates the theoretical optimum for binary classification tasks, preventing class imbalance bias while preserving maximum sample diversity. The balancing procedure prioritized retention of all hate samples containing human rationales (from HateXplain) while strategically undersampling normal content to achieve the target distribution.

### Rationale Coverage and Interpretability

A distinguishing characteristic of this unified dataset is enhanced rationale coverage achieved through the balancing strategy. While the original unbalanced corpus exhibited only 3.2% rationale coverage (2,726 HateXplain samples among 64,321 total), the balanced dataset achieves 36.9% coverage (1,898 samples with human explanations among 5,151 total)—representing an 11.5-fold improvement. These rationales were synthesized from HateXplain's token-level annotations into natural language explanations following the template: "The words '{flagged_tokens}' were flagged because they {harm_description} targeting {target_group}."

### Data Partitioning

The dataset maintains the original train/validation/test split ratios from the source corpora to preserve temporal and distributional characteristics:

- **Training set**: 3,628 samples (70.4%)
- **Validation set**: 514 samples (10.0%)
- **Test set**: 1,009 samples (19.6%)

Within each partition, source distribution balance is maintained (approximately 47% HateXplain, 53% ToxiGen), ensuring consistent representation across training, hyperparameter tuning, and final evaluation phases. All sampling procedures employed a fixed random seed (42) to ensure experimental reproducibility.

### Quality Metrics and Characteristics

The unified dataset exhibits the following empirical characteristics:

- **Average text length**: 102.1 characters (supporting both short-form social media posts and longer contextualized statements)
- **Source reduction**: 92% reduction from original 64,321 samples, prioritizing quality over quantity through strategic balancing
- **Label mapping consistency**: Binary labels derived from HateXplain's `hatespeech`→`hate` and `offensive`/`normal`→`normal`; ToxiGen's `prompt_label=1`→`hate` and `prompt_label=0`→`normal`
- **Metadata preservation**: Original sample IDs, split assignments, and source dataset provenance retained for audit trail completeness

### Unification Rationale

The decision to unify HateXplain and ToxiGen addresses a fundamental limitation in single-source hate speech datasets: the inability to capture both explicit and implicit hate modalities within a single training corpus. HateXplain's strength in persona-based explicit targeting complements ToxiGen's focus on subtle, context-dependent toxicity. This complementary coverage enables model training on diverse hate speech manifestations, potentially improving generalization to real-world content that exhibits varying degrees of explicitness.

Furthermore, the severe source imbalance in the original unbalanced corpus (95.8% ToxiGen dominance) would bias model learning toward implicit hate patterns while underrepresenting persona-based targeting. The implemented 47:53 source balance ensures neither modality dominates the training signal, facilitating robust feature learning across both hate speech types.

## Dataset Availability and Reproducibility

All data preparation procedures are implemented in `data_preparation/data_unification.py` with complete documentation in `data_preparation/UNIFICATION_APPROACH.md`. The unified dataset is persisted in `data/processed/unified_dataset_final.json` with accompanying train/validation/test files. The stratified sampling procedure employs deterministic random seeding (seed=42), ensuring bitwise-identical dataset reconstruction across experimental replications.

This unified dataset serves as the experimental foundation for all prompt engineering evaluations, instruction fine-tuning experiments, and model performance assessments presented in this thesis.
