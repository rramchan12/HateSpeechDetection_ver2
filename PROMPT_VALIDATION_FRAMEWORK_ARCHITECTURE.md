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
The Template Loader subsystem (`loaders/strategy_templates_loader.py`) implements a declarative JSON-based approach to prompt strategy management, enabling researchers to define, modify, and evaluate different prompting strategies across multiple AI models without code modification. This design philosophy supports comparative model evaluation (GPT-5 vs. GPT-OSS) while maintaining experimental rigor through version-controlled configurations.

### Template Loader Component Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                     TEMPLATE LOADER SUBSYSTEM                                │
│                  (strategy_templates_loader.py)                              │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │   JSON Configuration Files           │
              │   (prompt_templates/ directory)      │
              └──────────┬───────────────────────────┘
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │   BASELINE   │ │   COMBINED   │ │ ALL_COMBINED │
  │  TEMPLATES   │ │  TEMPLATES   │ │   (Master)   │
  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │  GPT-5  │      │  GPT-5  │      │ All     │
    │baseline_│      │combined_│      │Strategies│
    │v1_gpt5  │      │gpt5_v1  │      │(5 total)│
    └─────────┘      └─────────┘      └─────────┘
    ┌─────────┐      ┌─────────┐
    │ GPT-OSS │      │ GPT-OSS │
    │baseline_│      │combined_│
    │v1       │      │gptoss_v1│
    └─────────┘      └─────────┘

                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     STRATEGY REGISTRY                  │
        │  Dict[str, PromptStrategy]             │
        │                                        │
        │  Key-Value Store for O(1) Lookup:     │
        │  ├─ "baseline_conservative" → Strategy│
        │  ├─ "baseline_standard" → Strategy    │
        │  ├─ "combined_optimized" → Strategy   │
        │  ├─ "combined_focused" → Strategy     │
        │  └─ "combined_conservative" → Strategy│
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │   RUNTIME INSTANTIATION                │
        │                                        │
        │  For each sample:                      │
        │  1. Retrieve strategy from registry   │
        │  2. Format template with {text}       │
        │  3. Substitute {target_group}         │
        │  4. Return complete prompt + params   │
        └────────────────────────────────────────┘
```

### Template Structure Specification

Each JSON template file follows a standardized hierarchical structure designed for declarative prompt engineering:

```text
TEMPLATE FILE STRUCTURE
═══════════════════════════════════════════════════════════════

{
  "strategies": {                          ← Top-level container
    
    "strategy_name": {                     ← Unique strategy identifier
      
      ┌─────────────────────────────────────────────────────┐
      │ 1. METADATA LAYER                                   │
      ├─────────────────────────────────────────────────────┤
      │ "name": "strategy_name"                             │
      │ "description": "Strategy purpose and approach..."   │
      └─────────────────────────────────────────────────────┘
      
      ┌─────────────────────────────────────────────────────┐
      │ 2. SYSTEM PROMPT LAYER                              │
      │    (Role Definition & Structural Constraints)       │
      ├─────────────────────────────────────────────────────┤
      │ "system_prompt": "You are a content moderation...   │
      │                                                     │
      │ Components:                                         │
      │ ├─ Role Definition: Model's identity/capabilities  │
      │ ├─ Response Format: JSON structure requirements    │
      │ ├─ Classification Rules: Label constraints         │
      │ ├─ Reasoning Instructions: Analysis approach       │
      │ └─ Detection Focus: What to prioritize            │
      └─────────────────────────────────────────────────────┘
      
      ┌─────────────────────────────────────────────────────┐
      │ 3. USER TEMPLATE LAYER                              │
      │    (Parameterized Input with Runtime Substitution)  │
      ├─────────────────────────────────────────────────────┤
      │ "user_template": "Policy context...                 │
      │                                                     │
      │                  Text: \"{text}\"                   │
      │                                                     │
      │                  Community focus for {target_group}│
      │                  ..."                               │
      │                                                     │
      │ Placeholders:                                       │
      │ ├─ {text}: Sample text for classification          │
      │ └─ {target_group}: Demographic context (optional)  │
      └─────────────────────────────────────────────────────┘
      
      ┌─────────────────────────────────────────────────────┐
      │ 4. MODEL PARAMETERS LAYER                           │
      │    (Generation Hyperparameters)                     │
      ├─────────────────────────────────────────────────────┤
      │ "parameters": {                                     │
      │   "max_tokens": 200-1024,        ← Response length │
      │   "temperature": 0.0-1.0,        ← Randomness      │
      │   "top_p": 0.8-1.0,              ← Nucleus sampling│
      │   "frequency_penalty": 0.0-0.2,  ← Repetition ctrl │
      │   "presence_penalty": 0.0-0.1,   ← Topic diversity │
      │   "response_format": "json_object" ← Output format │
      │ }                                                   │
      └─────────────────────────────────────────────────────┘
    }
  }
}

INSTANTIATION FLOW (Runtime)
═══════════════════════════════════════════════════════════════

Template Definition (Static JSON)
         │
         ▼
   "{text}" placeholder
         │
         ▼
   Runtime Substitution
   str.format(text="actual sample text...")
         │
         ▼
   Complete Prompt
   "Classify the following text: actual sample text..."
         │
         ▼
   Submitted to Model API
   (with parameters: temperature=0.1, max_tokens=512, ...)
```

### Template Taxonomy: Baseline vs. Combined Strategies

The framework organizes prompt templates into two primary categories, each optimized for different AI models:

#### **BASELINE Templates** (Minimal Prompting Approach)

**Purpose**: Establish performance floor using minimal contextual guidance, relying on model's pre-trained knowledge of hate speech patterns without explicit policy frameworks or community perspectives.

**Characteristics**:
- Short, direct instructions (~150-200 words)
- No explicit policy text or community perspectives
- Relies on general understanding of "hate speech" and "social norms"
- Minimal token usage (200-350 tokens max)

**Model-Specific Variants**:

1. **GPT-5 Baseline Templates** (`baseline_v1_gpt5.json`):
   - Three variants optimized for GPT-5's enhanced reasoning:
     * `baseline_conservative`: Ultra-low temperature (1.0), 200 tokens, minimal overhead
     * `baseline_standard`: Balanced parameters (temp=1.0), 300 tokens, reliability focus
     * `baseline_balanced`: Production-optimized (temp=1.0), 350 tokens, comprehensive rationale
   - Leverages GPT-5's improved safety alignment and contextual understanding
   - Designed to test GPT-5's zero-shot hate speech detection without scaffolding

2. **GPT-OSS Baseline Templates** (`baseline_v1.json`):
   - Single baseline variant optimized for open-source models
   - Temperature: 0.1 (lower for more deterministic outputs on OSS models)
   - Max tokens: 512 (standard allocation)
   - Tests OSS model capabilities without advanced prompting techniques

**Experimental Value**: Establishes baseline performance metrics against which advanced prompting strategies (Combined) can be compared to quantify prompt engineering impact.

#### **COMBINED Templates** (Advanced Multi-Faceted Approach)

**Purpose**: Maximize detection accuracy and demographic fairness by integrating multiple complementary perspectives: (1) X platform's official hateful conduct policy for regulatory compliance, (2) community-informed harm analysis from LGBTQ+, Middle Eastern, and Mexican/Latino perspectives, and (3) few-shot examples demonstrating edge cases.

**Characteristics**:
- Extended, structured instructions (~800-1500 words)
- Explicit policy text from X platform's hateful conduct guidelines
- Community-specific harm assessment frameworks
- Few-shot examples for Mexican/Latino, LGBTQ+, and Middle Eastern contexts
- Cultural awareness guidance (in-group reclamation vs. out-group attacks)
- Coded/subtle hate detection emphasis
- Higher token allocation (400-650 tokens max)

**Model-Specific Variants**:

1. **GPT-5 Combined Templates** (`combined_gpt5_v1.json`):
   - Three temperature-optimized variants for GPT-5:
     * `combined_optimized`: Confidence-aware analysis (temp=1.0, 650 tokens)
       - Adaptive reasoning: high-confidence cases get direct classification, ambiguous cases get multi-perspective analysis
       - Explicit confidence scoring: {"classification": "hate", "confidence": "high/medium/low"}
       - Expanded few-shot examples with Mexican/Latino immigration hate detection focus
     * `combined_focused`: Cultural context integration (temp=1.0, 500 tokens)
       - Direct binary classification with cultural awareness framework
       - Historical context and power dynamics considerations
       - Balanced few-shot coverage across all three demographic groups
     * `combined_conservative`: Minimal overhead precision (temp=1.0, 400 tokens)
       - High-confidence scenario optimization
       - Condensed examples while maintaining core policy + community perspectives
   - All GPT-5 variants use temperature=1.0 (leveraging GPT-5's improved reasoning stability)

2. **GPT-OSS Combined Templates** (`combined_gptoss_v1.json`):
   - Three temperature-tuned variants for open-source models:
     * `combined_optimized`: Comprehensive approach (temp=0.1, 512 tokens)
       - Full policy text with few-shot examples
       - Detailed Mexican/Latino hate vs. policy discussion examples
       - LGBTQ+ in-group reclamation context
     * `combined_focused`: Balanced variant (temp=0.05, 256 tokens)
       - Restored LGBTQ+ cultural context
       - Focused few-shot examples
       - Lower temperature for deterministic outputs
     * `combined_conservative`: High-precision variant (temp=0.0, 256 tokens)
       - Zero temperature for maximum determinism
       - Conservative token allocation
       - Minimal but effective guidance
   - OSS variants use lower temperatures (0.0-0.1) to compensate for potentially higher output variance

**Experimental Value**: Tests hypothesis that explicit policy + community perspectives + cultural awareness significantly improves hate speech detection accuracy and reduces demographic bias compared to baseline prompting.

### Template Loading Process

**Phase 1: Initialization**
```python
loader = StrategyTemplatesLoader(templates_file_path)
├─ Resolve file path (default: prompt_templates/combined/all_combined.json)
├─ Initialize empty registry: strategies = {}
└─ Auto-invoke load_strategies()
```

**Phase 2: JSON Parsing & Object Construction**
```python
load_strategies()
├─ Open JSON file with UTF-8 encoding
├─ Parse JSON: data = json.load(f)
├─ Extract strategies dict: data["strategies"]
└─ For each strategy_name, strategy_config:
   ├─ Extract system_prompt string
   ├─ Extract user_template string
   ├─ Create PromptTemplate(system_prompt, user_template)
   ├─ Extract parameters dict
   ├─ Create PromptStrategy(name, description, template, parameters)
   └─ Register: strategies[strategy_name] = PromptStrategy(...)
```

**Phase 3: Runtime Strategy Retrieval & Instantiation**
```python
# Orchestration layer requests strategy
strategy = loader.strategies["combined_optimized"]  # O(1) lookup

# Format prompt with sample data
formatted_prompt = strategy.format_prompt(
    text="Sample text to classify...",
    target_group="lgbtq"
)
# Returns: Complete prompt with {text} and {target_group} substituted

# Get model parameters
params = strategy.get_model_parameters()
# Returns: {"max_tokens": 650, "temperature": 1.0, ...}

# Submit to model API
response = model_connector.complete(
    messages=[SystemMessage(system_prompt), UserMessage(formatted_prompt)],
    **params
)
```

### Design Rationale & Benefits

**Model-Agnostic Infrastructure**: Single loading mechanism supports both GPT-5 and GPT-OSS templates, enabling controlled comparative evaluation where template content is the experimental variable.

**Declarative Configuration**: Researchers modify JSON files to test new prompting strategies without touching Python code, reducing iteration time from minutes to seconds and minimizing bugs.

**Version Control & Reproducibility**: All templates are Git-versioned, enabling bitwise-identical prompt reconstruction across experimental replications and peer review validation.

**Temperature Optimization by Model Family**: GPT-5 templates use higher temperatures (1.0) leveraging improved reasoning stability, while GPT-OSS templates use lower temperatures (0.0-0.1) for deterministic outputs, reflecting model-specific tuning.

**Stratified Complexity**: Baseline templates establish performance floor (~200 words), Combined templates test ceiling (~800-1500 words), enabling quantification of prompt engineering impact through controlled comparison.

**Few-Shot Learning Integration**: Combined templates include targeted examples addressing known failure modes (Mexican/Latino immigration hate, LGBTQ+ in-group reclamation context), informed by error analysis from baseline validation runs.

This template loader architecture serves as the foundation for systematic prompt engineering experimentation, enabling rigorous evaluation of how prompt design choices (baseline vs. combined), cultural context integration, and model selection (GPT-5 vs. GPT-OSS) impact hate speech detection performance and demographic fairness.

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
