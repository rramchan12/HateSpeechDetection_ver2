# Data Preparation Implementation

## Overview

The data preparation module implements a three-stage pipeline transforming raw HateXplain and ToxiGen datasets into a unified, balanced corpus optimized for hate speech detection research. The implementation addresses complementary hate speech modalities—HateXplain's explicit persona-based attacks with human rationales versus ToxiGen's implicit, context-dependent toxicity—through strategic unification balancing source representation, class distribution, and interpretability signals. The module architecture separates dataset-specific preprocessing (`data_preparation_hatexplain.py`, `data_preparation_toxigen.py`) from cross-dataset unification logic (`data_unification.py`), enabling independent evolution of source-specific transformations while maintaining unified schema compliance.

## Implementation Architecture

### Stage 1: Source-Specific Preprocessing

The preprocessing stage normalizes heterogeneous source formats into intermediate representations conforming to dataset-specific schemas. HateXplain preprocessing (`data_preparation_hatexplain.py`) extracts annotation majority labels from multi-annotator judgments, consolidates token-level attention rationales into text explanations, and filters samples to three target demographics (LGBTQ+, Mexican/Latino, Middle Eastern) while mapping original categories (homosexual→lgbtq, hispanic→mexican, arab→middle_east). ToxiGen preprocessing (`data_preparation_toxigen.py`) loads HuggingFace-materialized Parquet splits, maps binary toxicity labels to hate/normal classification, normalizes target group identifiers for demographic consistency, and applies quality filters removing degenerate samples and malformed annotations. Both preprocessors maintain train/validation/test split integrity (70.4%/10.0%/19.6%) established during initial data collection, ensuring evaluation set isolation throughout the pipeline.

### Stage 2: Schema Unification and Balancing

The unification stage (`data_unification.py`) implements the `DatasetUnifier` class orchestrating cross-dataset integration through standardized field mapping and strategic sampling. The unified schema defines eight core fields: `text` (input content), `label_binary` (hate/normal classification), `label_multiclass` (source-specific granular labels), `target_group_norm` (standardized demographic identifier), `persona_tag` (top-level group categorization), `source_dataset` (provenance tracking), `rationale_text` (human explanations when available), and `is_synthetic` (generation indicator). The unifier implements Approach 3 from the comprehensive evaluation documented in `UNIFICATION_APPROACH.md`: HateXplain undersampling with label balance. This strategy preserves all valuable hate samples containing human rationales while strategically undersampling HateXplain normal instances, then matches ToxiGen samples at 1:1 source ratio through stratified sampling across target groups maintaining 50/50 hate/normal balance per demographic.

The balancing algorithm (`balance_source_distribution()` method) executes four operations: correcting HateXplain label mappings where 'hatespeech' and 'offensive' categories properly map to hate classification, undersampling HateXplain normal samples to achieve overall label balance while preserving all hate instances with rationales, stratified ToxiGen sampling proportional to target group distribution ensuring equal hate/normal representation per demographic, and split redistribution maintaining original train/validation/test ratios across the balanced corpus. The implementation employs deterministic random seeding (seed=42) ensuring reproducible sampling across experimental runs, critical for peer review validation and longitudinal research reproducibility.

### Stage 3: Export and Validation

The export stage serializes unified datasets to `data/processed/unified/` directory in multiple formats supporting diverse downstream workflows. JSON Lines format (`unified_train.jsonl`, `unified_val.jsonl`, `unified_test.jsonl`) provides human-readable text-based storage enabling inspection and version control integration. JSON format (`unified_train.json`, `unified_val.json`, `unified_test.json`) offers structured hierarchical representation suitable for programmatic loading without streaming parsers. The unifier generates comprehensive statistics (`unified_dataset_stats.json`) documenting final corpus characteristics: total sample count (5,151), source distribution (47.1% HateXplain, 52.9% ToxiGen), binary label balance (47.1% hate, 52.9% normal), rationale coverage (36.9%), and per-group demographic representation (LGBTQ+ 48.8%, Middle Eastern 28.5%, Mexican/Latino 22.6%).

## Key Implementation Features

The unification implementation achieves four critical objectives addressing methodological challenges in hate speech dataset curation. Source balance (47/53 HateXplain/ToxiGen ratio) ensures equal representation of explicit persona-based and implicit context-dependent hate modalities, preventing model bias toward either pattern type. Class balance (47/53 hate/normal distribution) optimizes training dynamics by avoiding majority-class prediction shortcuts while maintaining realistic prevalence ratios. Rationale preservation maximizes interpretability signals by retaining 100% of HateXplain samples containing human explanations, yielding 36.9% corpus-wide rationale coverage versus 3.2% in the unbalanced original—an 11.5× improvement enabling explainable AI research. Efficiency optimization reduces corpus size by 92% (64,321→5,151 samples) through strategic sampling eliminating redundant ToxiGen instances, accelerating training by approximately 12× while maintaining representative coverage of both hate modalities and all target demographics.

## Usage

**Preprocess individual datasets:**
```bash
python -m data_preparation.data_preparation_hatexplain
python -m data_preparation.data_preparation_toxigen
```

**Unify datasets with balancing:**
```bash
python -m data_preparation.data_unification --balance
```

**Output location:** `data/processed/unified/` containing train/val/test splits in JSONL and JSON formats plus comprehensive statistics file documenting corpus characteristics and balancing outcomes.

## Design Rationale

The selected unification approach (detailed analysis in `UNIFICATION_APPROACH.md`) represents optimal trade-offs among five candidate strategies evaluated across source balance, class distribution, rationale coverage, and training efficiency dimensions. Approach 3 (HateXplain undersampling with label balance) dominates alternatives by simultaneously achieving near-perfect source balance (47/53 vs target 50/50), optimal class distribution (47/53 hate/normal), maximum rationale preservation (36.9% coverage), and substantial efficiency gains (92% size reduction). Alternative approaches exhibited critical weaknesses: Approach 1 (no balancing) suffered severe source imbalance (4.2% HateXplain) and minimal rationale coverage (3.2%), Approach 2 (simple 1:1 balance) introduced label distribution skew requiring correction, while Approaches 4-5 (1:2, 1:3 ratios) sacrificed rationale coverage and source balance for marginal data volume increases offering diminishing returns. The implemented strategy establishes methodological rigor for multi-source hate speech corpus construction, providing replicable procedures for future dataset integration research.
