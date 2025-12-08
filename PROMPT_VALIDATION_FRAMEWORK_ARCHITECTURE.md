# Hate Speech Prompt Validation Framework: Architecture and Design

## Framework Overview

The Hate Speech Prompt Validation Framework implements a modular architecture for systematic evaluation of prompt engineering strategies in hate speech detection. The framework enables rigorous empirical assessment through declarative JSON/YAML configuration, supporting rapid experimentation without code modification. The design emphasizes reproducibility through deterministic sampling (seed=42), scalability via concurrent execution (ThreadPoolExecutor), and comprehensive analytics including classification metrics and demographic bias analysis.

## Architectural Block Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER (prompt_runner.py)           │
│  CLI Interface • Workflow Coordination • Concurrent Execution       │
│  Retry Logic • Rate Limit Handling • Result Aggregation            │
└────────┬─────────────┬─────────────┬─────────────┬─────────────────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
    ┌────────┐   ┌──────────┐  ┌──────────┐  ┌───────────────┐
    │ CONFIG │   │  MODEL   │  │ TEMPLATE │  │     DATA      │
    │  MGMT  │   │CONNECTOR │  │  LOADER  │  │  MANAGEMENT   │
    └────────┘   └──────────┘  └──────────┘  └───────────────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
    ┌────────────────────────────────────────────────────────┐
    │              INFERENCE ENGINE                          │
    │  Template Instantiation → Message Construction →       │
    │  API Request → Response Parse → Label Standardization  │
    └────────────────────────────────────────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │     PERSISTENCE SUBSYSTEM    │
               │  Incremental CSV • RunID Org │
               └──────────────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  METRICS COMPUTATION         │
               │  Classification • Bias • F1  │
               └──────────────────────────────┘
```

## Component Specifications

### 1. Orchestration Layer (`prompt_runner.py`)
Coordinates end-to-end validation through CLI interface with concurrent execution (ThreadPoolExecutor, 5 workers default, 10-sample batches). Implements exponential backoff retry (delay = min(30, 2^attempt + random(0,1))) and intelligent rate limiting via HTTP header monitoring (`x-ratelimit-remaining-requests/tokens`). Generates four output files per run: validation results CSV, test samples CSV, performance metrics CSV, and human-readable evaluation report.

### 2. Configuration Management (`connector/model_connection.yaml`)
YAML-based multi-model configuration with environment variable substitution (`${VAR_NAME}` syntax) for secure credential management. Supports arbitrary model deployments (GPT-OSS-120B, GPT-5, custom endpoints) with model-specific parameter defaults (temperature, max_tokens, top_p, response_format).

### 3. Model Connector (`connector/azureai_connector.py`)
Azure AI Inference SDK abstraction implementing `SystemMessage`/`UserMessage` construction and persistent HTTP connection pooling. Validates credentials pre-connection with comprehensive error handling for network failures, authentication errors, and rate limiting (HTTP 429).

### 4. Template Loader (`loaders/strategy_templates_loader.py`)
Parses JSON strategy configurations into `PromptStrategy` objects containing `PromptTemplate` (system_prompt, user_template) and generation parameters. Implements runtime variable substitution (`{text}`, `{target_group}`) via `str.format()` with O(1) strategy lookup from registry dictionary.

### 5. Data Management (`loaders/unified_dataset_loader.py`)
Stratified sampling across target groups (LGBTQ+ 48.8%, Middle Eastern 28.5%, Mexican/Latino 22.6%) and binary labels (47:53 hate/normal). Maintains train/val/test splits (70.4%/10.0%/19.6%) with deterministic seeding (seed=42). Provides curated test sets: `canned_50_quick`, `canned_100_size_varied`, `canned_100_stratified`.

### 6. Inference Engine (`prompt_runner.py` processing methods)
Six-stage pipeline: (1) template instantiation, (2) message construction, (3) API request with retry, (4) JSON response parsing (fallback to text extraction), (5) label standardization (hate/hateful/hate_speech→hate; normal/benign/not_hate→normal), (6) rationale extraction. Thread-pool parallelism with batch processing and per-sample error capture.

### 7. Persistence Subsystem (`metrics/persistence_helper.py`)
Incremental CSV streaming (`csv.DictWriter`) with immediate disk flush per sample, preventing memory accumulation. Timestamp-based run directories (`outputs/run_YYYYMMDD_HHMMSS/`) containing validation results, test samples, metrics, evaluation report, and execution logs. Preserves command-line arguments, template paths, and model configuration for reproducibility.

### 8. Metrics Computation (`metrics/evaluation_metrics_calc.py`)
Classification metrics via scikit-learn: accuracy (TP+TN)/(TP+TN+FP+FN), precision TP/(TP+FP), recall TP/(TP+FN), F1-score (harmonic mean). Bias analysis: per-group FPR (FP/(FP+TN)) and FNR (FN/(FN+TP)) for LGBTQ+, Middle Eastern, Mexican/Latino demographics. Supports post-hoc recalculation via `--metrics-only` flag.

## Execution Workflow

**Five-Phase Pipeline**:
1. **Initialization**: Parse CLI → Load YAML config (substitute env vars) → Load JSON strategies → Initialize Azure AI client
2. **Dataset Preparation**: Load unified/canned dataset → Apply stratified sampling → Validate schema (text, label_binary, target_group_norm)
3. **Validation Execution**: Create runID directory → Initialize CSV writer → Concurrent sample processing (template→message→API→parse→standardize→write)
4. **Metrics Computation**: Load results CSV → Compute accuracy/F1/precision/recall → Generate confusion matrices → Calculate per-group bias (FPR/FNR)
5. **Report Generation**: Write performance_metrics.csv → Write evaluation_report.txt → Archive logs

## Design Principles

**Modularity**: Cohesive subsystems with well-defined interfaces enable independent testing and evolution without cross-component dependencies.

**Declarative Configuration**: JSON strategy specification and YAML model configuration eliminate code modifications, accelerating research iteration cycles.

**Reproducibility**: Deterministic seeding (42), comprehensive metadata logging, and configuration versioning ensure experimental reproducibility.

**Scalability**: Concurrent execution and incremental persistence enable large-scale evaluations (thousands of samples, multiple strategies) within time/memory constraints.

**Resilience**: Exponential backoff, comprehensive error handling, and partial result preservation ensure graceful degradation under network failures and API rate limiting.

This framework serves as the experimental infrastructure for prompt engineering studies, enabling systematic evaluation across performance, bias, and interpretability dimensions.

---

## Deep Dive: Template Loader Architecture

### Overview
The Template Loader subsystem (`loaders/strategy_templates_loader.py`) implements a declarative approach to prompt strategy management, enabling researchers to define, modify, and test different prompting strategies without code modification. This design philosophy accelerates experimental iteration and ensures reproducibility through version-controlled JSON configurations.

### JSON Strategy Configuration Schema

Each strategy in the JSON configuration file (`prompt_templates/combined/all_combined.json`) contains five essential components:

1. **Strategy Metadata**:
   - `name`: Unique identifier for runtime strategy selection (e.g., "baseline", "combined")
   - `description`: Human-readable explanation of the strategy's approach and purpose

2. **System Prompt**: Role definition and structural constraints that establish the model's behavior, response format requirements (strict JSON), and classification rules

3. **User Template**: Parameterized prompt containing placeholders (`{text}`, `{target_group}`) that are dynamically substituted with sample-specific values during runtime

4. **Model Parameters**: Generation hyperparameters controlling response characteristics:
   - `temperature`: Randomness control (0.05-0.2, lower for deterministic classification)
   - `max_tokens`: Response length limit (512-1024 tokens)
   - `top_p`: Nucleus sampling threshold (0.9-1.0)
   - `frequency_penalty`: Repetition reduction (0.0-0.2)
   - `presence_penalty`: Topic diversity (0.0-0.1)
   - `response_format`: Output structure specification ("json_object")

### Loading Process

**Initialization Phase**:
```
StrategyTemplatesLoader.__init__()
├─ Resolve template file path (default: prompt_templates/combined/all_combined.json)
├─ Initialize empty strategies dictionary: Dict[str, PromptStrategy]
└─ Invoke load_strategies() to parse JSON and populate registry
```

**JSON Parsing and Validation**:
```python
load_strategies()
├─ Read JSON file with UTF-8 encoding
├─ Parse JSON structure: {"strategies": {...}}
├─ For each strategy configuration:
│  ├─ Extract system_prompt and user_template strings
│  ├─ Create PromptTemplate(system_prompt, user_template)
│  ├─ Extract parameters dictionary
│  └─ Construct PromptStrategy object with all components
└─ Store in registry: strategies[name] = PromptStrategy(...)
```

**Runtime Template Instantiation**:
When processing each sample, the orchestration layer:
1. Retrieves strategy from registry via O(1) dictionary lookup
2. Calls `strategy.format_prompt(text=sample_text, target_group=group)`
3. Substitutes placeholders using Python's `str.format()` method
4. Returns fully instantiated prompt ready for model submission

### Strategy Diversity

The framework supports five distinct prompting strategies with varying complexity and theoretical foundations:

- **Baseline**: Minimal approach relying on general understanding without explicit policy or community context (temperature=0.1, max_tokens=512)
- **Policy**: X platform's hateful conduct policy with explicit violation criteria (temperature=0.2, max_tokens=768)
- **Persona**: Multi-perspective analysis from LGBTQ+, Middle Eastern, and Mexican/Latino community viewpoints (temperature=0.1, max_tokens=768)
- **Combined**: Unified policy + persona approach with both regulatory compliance and community harm assessment (temperature=0.1, max_tokens=768)
- **Enhanced Combined**: Optimized variant with expanded guidance, explicit slur examples, and systematic analysis framework (temperature=0.05, max_tokens=1024)

### Design Benefits

**Reproducibility**: JSON configurations are version-controlled, enabling bitwise-identical prompt reconstruction across experimental replications

**Rapid Iteration**: New strategies can be added by editing JSON without Python code changes, reducing iteration time from minutes to seconds

**Comparative Analysis**: Identical infrastructure processes all strategies, isolating prompt design as the only experimental variable

**Parameter Optimization**: Model parameters are strategy-specific, allowing temperature/sampling tuning per prompting approach

---

## Deep Dive: Data Management Architecture

### Overview
The Data Management subsystem (`loaders/unified_dataset_loader.py`) provides flexible, reproducible dataset access with support for both large-scale unified test sets and curated canned samples. The design emphasizes stratified sampling, deterministic reproducibility, and efficient caching.

### Data Source Architecture

**Unified Test Dataset** (`data/processed/unified/unified_test.json`):
- **Size**: 1,010 samples from held-out test split (19.6% of 5,151 total)
- **Composition**: 47.1% hate, 52.9% normal (balanced binary labels)
- **Demographics**: LGBTQ+ (48.8%), Middle Eastern (28.5%), Mexican/Latino (22.6%)
- **Source Mix**: HateXplain (47.1%), ToxiGen (52.9%)
- **Rationale Coverage**: 36.9% of samples include human-annotated explanations
- **Purpose**: Comprehensive evaluation on data not seen during prompt development

**Canned Test Sets** (`prompt_engineering/data_samples/`):
Three pre-curated subsets optimized for specific validation scenarios:

1. **canned_50_quick.json** (50 samples):
   - Purpose: Rapid iteration and debugging
   - Use case: Quick validation during prompt development
   - Sampling: Stratified by label and target group

2. **canned_100_size_varied.json** (100 samples):
   - Purpose: Text length sensitivity testing
   - Use case: Evaluating prompt robustness across short/medium/long texts
   - Sampling: Diverse character counts (20-500+ characters)

3. **canned_100_stratified.json** (100 samples):
   - Purpose: Balanced mini-evaluation
   - Use case: Representative quick assessment before full validation
   - Sampling: Proportional demographic and label distribution

### Sample Data Structure

Each sample in all datasets contains standardized fields:

```json
{
  "text": "Sample text content for classification",
  "label_binary": "hate" | "normal",
  "label_multiclass": "hatespeech" | "offensive" | "normal",
  "target_group_norm": "lgbtq" | "middle_eastern" | "mexican",
  "persona_tag": "homosexual" | "muslim" | "hispanic",
  "source_dataset": "hatexplain" | "toxigen",
  "is_synthetic": false | true,
  "rationale_text": "Human-annotated explanation (if available)",
  "original_id": "Unique identifier from source dataset",
  "split": "train" | "val" | "test"
}
```

### Loading and Sampling Process

**Initialization with Caching**:
```
UnifiedDatasetLoader.__init__()
├─ Define dataset paths (canned_path, unified_path)
├─ Initialize cache dictionaries (_canned_cache, _unified_cache)
└─ Configure logger for diagnostic output
```

**Lazy Loading with Cache Optimization**:
```python
load_samples(dataset_type, num_samples, random_seed)
├─ Check cache: if dataset loaded previously, return cached version
├─ If cache miss:
│  ├─ Read JSON file from disk
│  ├─ Parse JSON structure (list or {"samples": [...]})
│  ├─ Store in cache for future requests
│  └─ Log loading statistics
├─ Apply sample filtering:
│  ├─ If num_samples="all": return complete dataset
│  └─ If num_samples=N (integer):
│     ├─ Set random seed (default: None, reproducible if specified)
│     ├─ Apply random.sample(samples, N)
│     └─ Return stratified subset
└─ Return List[Dict[str, Any]]
```

### Stratified Sampling Implementation

While the current implementation uses `random.sample()` for simplicity, the unified dataset itself was created with stratified sampling ensuring balanced representation:

- **Label Balance**: 47:53 hate/normal ratio maintained across all demographic groups
- **Demographic Proportions**: LGBTQ+ (48.8%), Middle Eastern (28.5%), Mexican/Latino (22.6%) preserved in all subsets
- **Source Balance**: HateXplain (47.1%) and ToxiGen (52.9%) proportional distribution
- **Rationale Distribution**: 36.9% rationale coverage evenly distributed across groups

### Reproducibility Mechanisms

**Deterministic Seeding**:
The framework supports reproducible sampling via the `random_seed` parameter:
```python
loader.load_samples(dataset_type="unified", num_samples=100, random_seed=42)
```
This ensures identical sample selection across experimental replications, enabling:
- Bitwise-identical result reproduction
- Valid performance comparisons between strategy iterations
- Peer review and result verification

**Dataset Versioning**:
All dataset files are version-controlled in Git, preserving:
- Original data unification parameters (seed=42)
- Canned sample selections
- Test split assignments

### Usage Patterns

**Development Mode** (fast iteration):
```python
samples = loader.load_samples("canned", num_samples=50, random_seed=42)
```

**Comprehensive Evaluation** (full test set):
```python
samples = loader.load_samples("unified", num_samples="all")
```

**Stratified Quick Test** (representative subset):
```python
samples = loader.load_samples("canned", num_samples=100, random_seed=42)
```

### Integration with Orchestration Layer

The orchestration layer (`prompt_runner.py`) invokes the data loader during the Dataset Preparation Phase:

```
CLI Argument: --data-source unified --sample-size 200
         ↓
UnifiedDatasetLoader.load_samples("unified", 200, seed=42)
         ↓
Returns: 200 randomly sampled test instances
         ↓
Validation Execution Phase processes each sample
```

This architecture separates data management concerns from prompt validation logic, enabling independent evolution of dataset curation strategies while maintaining stable validation infrastructure.
