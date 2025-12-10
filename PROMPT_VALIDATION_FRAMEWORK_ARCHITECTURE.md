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
The Template Loader subsystem (`loaders/strategy_templates_loader.py`) implements a declarative JSON-based approach to prompt strategy management. Researchers can define and modify prompting strategies through JSON configuration files without code modification, supporting comparative evaluation across multiple AI models (GPT-5 vs. GPT-OSS) while maintaining experimental reproducibility through version control.

### Template Loader Component Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│              TEMPLATE LOADER SUBSYSTEM                             │
│           (strategy_templates_loader.py)                           │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │   JSON Configuration Files           │
        │   (prompt_templates/ directory)      │
        └──────┬──────────────┬────────────────┘
               │              │
         ┌─────┴─────┐   ┌────┴─────┐
         ▼           ▼   ▼          ▼
    ┌─────────┐  ┌─────────┐  ┌──────────┐
    │BASELINE │  │BASELINE │  │ COMBINED │  
    │ GPT-5   │  │ GPT-OSS │  │GPT-5/OSS │
    └─────────┘  └─────────┘  └──────────┘
         │              │            │
         └──────────────┴────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   STRATEGY REGISTRY          │
         │   Dict[str, PromptStrategy]  │
         │                              │
         │   O(1) Lookup:               │
         │   • baseline_conservative    │
         │   • baseline_standard        │
         │   • combined_optimized       │
         │   • combined_focused         │
         │   • combined_conservative    │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  RUNTIME INSTANTIATION       │
         │  1. Retrieve strategy        │
         │  2. Substitute {text}        │
         │  3. Substitute {target_group}│
         │  4. Return prompt + params   │
         └──────────────────────────────┘
```

### Template Structure Specification

Each JSON template contains four layers:

```json
{
  "strategies": {
    "strategy_name": {
      
      // 1. METADATA LAYER
      "name": "strategy_name",
      "description": "Strategy purpose and approach",
      
      // 2. SYSTEM PROMPT LAYER (Role & Constraints)
      "system_prompt": "You are a content moderation assistant...\n
                        - Response format: JSON\n
                        - Classification rules: hate/normal\n
                        - Detection focus: slurs, stereotypes, attacks",
      
      // 3. USER TEMPLATE LAYER (Parameterized Input)
      "user_template": "Policy context...\n
                        Text: \"{text}\"\n
                        Target group: {target_group}\n
                        Classify as hate or normal.",
      
      // 4. PARAMETERS LAYER (Model Hyperparameters)
      "parameters": {
        "max_tokens": 200-650,           // Response length
        "temperature": 0.0-1.0,          // Randomness control
        "top_p": 0.8-1.0,                // Nucleus sampling
        "frequency_penalty": 0.0-0.2,    // Repetition control
        "presence_penalty": 0.0-0.1,     // Topic diversity
        "response_format": "json_object" // Output format
      }
    }
  }
}
```

**Runtime Instantiation Flow**:
```
Static JSON Template → Extract {text} placeholder → 
Substitute with "actual sample text..." → 
Complete Prompt → Submit to Model API with parameters
```

### Template Taxonomy

The framework organizes templates into two categories across two model families:

**Template Categories**:

1. **BASELINE Templates**: Minimal prompting (~150-200 words)
   - Purpose: Establish performance floor without explicit guidance
   - Characteristics: No policy text, no community perspectives, general hate speech understanding
   - Token allocation: 200-350 tokens

2. **COMBINED Templates**: Advanced multi-faceted prompting (~800-1500 words)
   - Purpose: Maximize accuracy and fairness through policy + community perspectives + few-shot examples
   - Characteristics: X platform policy, LGBTQ+/Middle Eastern/Mexican Latino perspectives, cultural awareness, coded hate detection
   - Token allocation: 400-650 tokens

**Model-Specific Optimization**:

- **GPT-5 Templates**: 
  - Temperature: 1.0 (leveraging improved reasoning stability)
  - Variants: 3 baseline (conservative/standard/balanced), 3 combined (optimized/focused/conservative)
  
- **GPT-OSS Templates**: 
  - Temperature: 0.0-0.1 (compensating for output variance)
  - Variants: 1 baseline, 3 combined (optimized/focused/conservative)

### Loading Process

**Initialization → Parsing → Runtime Access**:

```python
# Phase 1: Initialization
loader = StrategyTemplatesLoader(templates_file_path)
# Resolves path, initializes registry, loads strategies

# Phase 2: JSON Parsing
load_strategies()
# Reads JSON → Parses {"strategies": {...}} → 
# Creates PromptTemplate + PromptStrategy objects →
# Stores in registry: strategies[name] = PromptStrategy(...)

# Phase 3: Runtime Retrieval
strategy = loader.strategies["combined_optimized"]  # O(1) lookup
formatted_prompt = strategy.format_prompt(
    text="Sample text...", 
    target_group="lgbtq"
)
params = strategy.get_model_parameters()

# Submit to model
response = model_connector.complete(
    messages=[SystemMessage(system_prompt), UserMessage(formatted_prompt)],
    **params
)
```

### Design Benefits

**Declarative Configuration**: JSON-based template definition enables rapid experimental iteration without code changes, reducing development cycle from minutes to seconds.

**Model-Agnostic Architecture**: Single loading mechanism supports both GPT-5 and GPT-OSS templates, isolating prompt design as the experimental variable for controlled comparative evaluation.

**Reproducibility**: Git-versioned JSON configurations ensure bitwise-identical prompt reconstruction across experimental replications and enable peer review validation.

**Stratified Complexity Testing**: Baseline templates (~200 words) establish performance floor, Combined templates (~800-1500 words) test performance ceiling, quantifying prompt engineering impact through controlled comparison.

**Parameter Optimization**: Model-specific temperature tuning (GPT-5: 1.0, GPT-OSS: 0.0-0.1) and token allocation (baseline: 200-350, combined: 400-650) optimized per model family characteristics.

This architecture provides the foundation for systematic evaluation of prompt design choices (baseline vs. combined), cultural context integration, and model selection (GPT-5 vs. GPT-OSS) on hate speech detection performance and demographic fairness. Subsequent sections detail individual template strategies and their empirical performance.

---

## GPT-OSS Baseline Prompting Strategies: A Parameter-Driven Exploration

### Overview

The GPT-OSS baseline framework implements systematic hyperparameter exploration to establish performance boundaries for open-source language models in hate speech detection. Maintaining identical prompt content across all variants, the framework investigates five distinct parameterization strategies that probe trade-offs between determinism and diversity, conciseness and comprehensiveness, and precision and recall. This controlled design isolates hyperparameter optimization as the independent variable, enabling rigorous assessment of how generation settings influence classification accuracy and demographic bias patterns.

### Architectural Framework

```text
┌──────────────────────────────────────────────────────────────┐
│         GPT-OSS BASELINE STRATEGY ARCHITECTURE               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
      ┌──────────────────────────────────────┐
      │   SHARED PROMPT FOUNDATION           │
      │   • Binary classification (hate/normal)
      │   • JSON format requirement          │
      │   • 1-2 sentence rationale          │
      └──────────────┬───────────────────────┘
                     │
         ┌───────────┼──────────┬──────────┬──────────┐
         ▼           ▼          ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
   │CONSERV- │ │STANDARD │ │CREATIVE│ │FOCUSED │ │EXPLORAT- │
   │ ATIVE   │ │         │ │        │ │        │ │ ORY      │
   └────┬────┘ └────┬────┘ └───┬────┘ └───┬────┘ └────┬─────┘
        │           │          │          │           │
   temp=0.0    temp=0.1   temp=0.3   temp=0.05   temp=0.5
   tok=256     tok=512    tok=768    tok=200     tok=1024
   top_p=0.9   top_p=1.0  top_p=0.95 top_p=0.8   top_p=0.85
   freq=0.0    freq=0.0   freq=0.1   freq=0.2    freq=0.3
   pres=0.0    pres=0.0   pres=0.1   pres=0.0    pres=0.2
```

### Parameterization Strategies

The Conservative strategy implements maximum determinism through zero-temperature sampling (0.0), eliminating stochastic variation in model outputs. With top-p at 0.9 and token allocation of 256, the strategy enforces greedy decoding that consistently selects highest-probability tokens, targeting production scenarios requiring reproducible classifications and minimal computational overhead. Both frequency and presence penalties remain at zero, prioritizing natural language patterns over diversity enforcement.

The Standard strategy establishes a balanced baseline through moderate temperature (0.1) that introduces minimal stochasticity while maintaining high determinism. Token allocation doubles to 512, accommodating comprehensive rationale generation. Maximum top-p (1.0) allows full probability distribution consideration while temperature constrains selection to high-probability regions. This configuration serves as the experimental control, representing conventional best practices for classification tasks.

The Creative strategy introduces controlled stochasticity (temperature 0.3) to probe nuanced reasoning in ambiguous cases. Token allocation expands to 768, hypothesizing that higher-temperature generation benefits from extended rationale space. Nucleus sampling at 0.95 truncates low-probability tokens while preserving diversity. Introduction of frequency penalty (0.1) and presence penalty (0.1) discourages repetitive phrasing, targeting scenarios where cultural context or coded language requires interpretive flexibility.

The Focused strategy optimizes for high-confidence classifications through aggressive constraints on diversity and output length. Temperature reduction to 0.05 approaches deterministic behavior with minimal stochastic variation. Token allocation constrains to 200, enforcing extreme conciseness under the hypothesis that clear cases require minimal justification. Top-p tightens to 0.8, and frequency penalty elevates to 0.2, maximizing information density for real-time moderation scenarios prioritizing latency minimization.

The Exploratory strategy maximizes generation diversity to discover novel hate speech patterns. Temperature elevation to 0.5 enables substantial stochastic variation, with maximum token allocation (1024) providing extensive reasoning space. Top-p at 0.85 balances diversity with quality. Combined frequency penalty (0.3) and presence penalty (0.2) represent aggressive diversity enforcement, actively discouraging repetition. This parameterization serves as a discovery mechanism, potentially revealing failure modes and bias patterns masked by conservative configurations.

### Experimental Rationale

This five-strategy framework operationalizes theoretical insights from large language model optimization research. The temperature dimension tests the determinism-diversity trade-off, where lower temperatures yield consistent but potentially brittle classifications, while higher temperatures enable nuanced reasoning at the cost of reproducibility. Token allocation investigates whether hate speech detection benefits from extensive rationale generation or if constrained outputs suffice. Frequency and presence penalties probe linguistic diversity's role in classification quality, testing whether diverse phrasing correlates with robust reasoning. Performance evaluation across demographic subgroups (LGBTQ+, Middle Eastern, Mexican/Latino) quantifies whether parameter choices introduce or mitigate bias patterns, addressing fairness concerns in automated content moderation.

---

## GPT-5 Baseline Prompting Strategies: Leveraging Enhanced Model Capabilities

### Overview

The GPT-5 baseline framework explores minimal prompting approaches optimized for next-generation language models with enhanced reasoning stability. Unlike GPT-OSS strategies requiring extensive hyperparameter tuning, GPT-5 investigates three parameterization strategies leveraging architectural improvements to achieve robust hate speech detection with simplified configuration. This controlled exploration maintains identical prompt content while testing whether GPT-5's enhanced capabilities enable effective classification through uniform temperature (1.0) and varied token allocation.

### Architectural Framework

```text
GPT-5 BASELINE STRATEGIES
         │
    ┌────┴────┐
    │ SHARED  │  Role: Content moderation
    │ PROMPT  │  Task: Binary classification
    └────┬────┘  Format: JSON + rationale
         │
    ┌────┼────┐
    ▼    ▼    ▼
  CONS  STD  BAL
  t=1.0 t=1.0 t=1.0    Unified temp=1.0
  200   300   350      Token variation
```

### Parameterization Strategies

The Conservative strategy establishes minimal viable configuration, employing temperature 1.0 with constrained token allocation of 200. This tests whether GPT-5's enhanced reasoning enables accurate classification with concise rationale generation, eliminating ultra-low temperatures required by earlier models. The strategy targets high-throughput production scenarios where computational efficiency is prioritized, leveraging GPT-5's deterministic tendencies at temperature 1.0 to maintain consistency without explicit temperature suppression.

The Standard strategy represents balanced experimental control, maintaining temperature 1.0 while expanding token allocation to 300. This serves as primary benchmark for assessing GPT-5's zero-shot detection capabilities, providing sufficient rationale space without verbose explanations. The moderate token budget accommodates GPT-5's tendency toward comprehensive yet concise outputs, reflecting optimization for clarity. This configuration tests conventional best practices adapted for next-generation models, where architectural improvements eliminate aggressive temperature constraints observed in GPT-OSS configurations.

The Balanced strategy optimizes production deployment through token allocation at 350 while maintaining unified temperature 1.0. This reflects empirical insights suggesting GPT-5's enhanced contextual understanding benefits from extended rationale space for nuanced reasoning in ambiguous cases, particularly coded language and implicit hate speech. The marginal token increase (17% over Standard) provides additional reasoning capacity without excessive computational overhead, isolating token allocation as the primary experimental variable.

### Experimental Rationale

The three-strategy framework fundamentally differs from GPT-OSS's five-strategy approach by leveraging architectural improvements that eliminate extensive hyperparameter exploration. The unified temperature 1.0 configuration reflects that GPT-5's enhanced reasoning stability produces consistent outputs at default sampling parameters, obviating determinism-diversity trade-offs central to GPT-OSS optimization. This enables focused investigation of token allocation's impact, testing whether hate speech detection benefits from minimal (200), moderate (300), or optimal (350) rationale lengths. The absence of frequency and presence penalty tuning reflects GPT-5's improved generation capabilities, where pre-training on diverse data reduces repetition without explicit penalty enforcement. Performance evaluation focuses on whether GPT-5's enhancements translate into superior accuracy and reduced demographic bias (LGBTQ+, Middle Eastern, Mexican/Latino) compared to GPT-OSS baselines, addressing critical fairness concerns in automated content moderation.

---

## Combined GPT-OSS Prompting Strategies: Iterative Noise-Reduction Discovery

### Overview

The Combined GPT-OSS framework investigates whether advanced prompt engineering can improve upon baseline performance through five iterative experimental cycles. After four failed iterations (V1-V4) where verbose additions degraded performance 4-28%, the fifth iteration achieved breakthrough through noise reduction. The framework systematically tests policy guidance, example-based learning, cultural context, and reasoning frameworks across escalating complexity levels to identify optimal signal compression strategies.

### Architectural Framework

```text
COMBINED GPT-OSS: 5-ITERATION EVOLUTION
         
V1: Policy + Examples (Verbose)           V2: Cultural Context           V3: Refined Examples
├─ combined_optimized                     ├─ cultural_context           ├─ recall_focused
├─ combined_focused                       ├─ recall_optimized           ├─ cultural_aware
├─ combined_conservative                  ├─ policy_focused             ├─ optimized
                                          ├─ persona_balanced
                                          ├─ minimal_hybrid

V4: Minimal Enhancements                  V5: NOISE-REDUCED (Breakthrough)
├─ minimal_examples                       ├─ implicit_examples ★ 
├─ balanced_lite                          ├─ chain_of_thought ★
├─ community_aware                        ├─ minimal_signal
├─ policy_lite                            ├─ compressed_tokens
├─ subtle_emphasis                        ├─ example_only

All configurations: temp=0.1, tokens=512, top_p=1.0
```

### Strategy Taxonomy

**V1 (Combined Policy + Persona with Examples)**: Fifteen examples across three demographic groups (five hate, five normal per LGBTQ+, Mexican/Latino, Middle Eastern) combined with 200-word X Platform Hateful Conduct Policy and structured evaluation framework. Tests whether comprehensive policy guidance plus extensive examples improves classification through explicit instruction.

**V2 (Cultural Context & Example Optimization)**: Five strategies testing zero to two examples with cultural awareness frameworks. Explores whether deep demographic context (cultural_context), recall optimization (recall_optimized), policy-heavy approaches (policy_focused), hybrid two-example configurations (persona_balanced), or minimal guidance (minimal_hybrid) outperform V1's verbose fifteen-example approach.

**V3 (Restored Examples with Refined Guidance)**: Returns to five examples per group after V2's zero-to-two approach failed. Tests whether V2's refined cultural guidance combined with V1's five-example structure improves performance through recall emphasis (recall_focused), cultural depth (cultural_aware), or balanced policy-persona frameworks (optimized).

**V4 (Minimal Baseline Enhancements)**: Explores the "goldilocks zone" between baseline simplicity and V1-V3 verbosity through six-example configurations (minimal_examples, balanced_lite), brief context additions (community_aware), compressed fifty-word policy (policy_lite), and single-sentence emphasis (subtle_emphasis). Tests whether minimal additions avoid V1-V3's degradation.

**V5 (Noise-Reduced Approaches)**: Five compression strategies testing demonstration over explanation. Implicit_examples presents six contrasting example pairs without policy text. Chain_of_thought implements four-step reasoning framework (identify attacks, check coded language, distinguish critique from hate, classify). Minimal_signal adds single-sentence policy definition. Compressed_tokens uses token-style markers [Policy: X]. Example_only presents raw examples without instructional framing.

### Experimental Rationale

The Combined framework systematically explores the complexity-performance relationship across five iterations. V1 tests conventional prompt engineering wisdom that comprehensive guidance improves performance through extensive examples and policy definitions. V2 investigates whether example reduction (zero to two) with deep cultural context avoids V1's potential information overload. V3 validates whether V1's five-example approach combined with V2's refined guidance achieves optimal balance. V4 explores minimal enhancements hypothesis that small additions to baseline avoid degradation while providing benefits. V5 tests noise-reduction through signal compression, replacing verbose explanations with implicit examples or structured reasoning frameworks.

The framework evolution reveals that verbosity consistently degrades performance regardless of implementation approach (policy guidance, examples, cultural context, persona instructions). Success emerges only when signals are compressed to 60-90 words through demonstration (implicit examples) or structured logic (chain-of-thought), eliminating explanatory text that conflicts with model pre-training or introduces ambiguous decision boundaries. This validates that effective instruction fine-tuning requires ruthless compression prioritizing high-density pattern encoding over comprehensive explanations.

---

## Combined GPT-5 Prompting Strategies: Architectural Optimization Under API Constraints

### Overview

The Combined GPT-5 framework addresses GPT-5's fixed temperature constraint (1.0) through architectural prompt engineering rather than hyperparameter tuning. Synthesizing GPT-5's hybrid adaptive reasoning with few-shot learning strategies proven effective in GPT-OSS combined approaches, the framework explores three architectural variants optimized for next-generation model capabilities. Unlike GPT-OSS combined strategies that leverage temperature manipulation, GPT-5 combined strategies optimize through reasoning structure (adaptive multi-stage vs. direct binary), cultural context integration depth, and strategic few-shot example allocation.

### Architectural Framework

```text
COMBINED GPT-5: ARCHITECTURAL OPTIMIZATION
         
         GPT-5 API Constraint: temp=1.0 (fixed)
         Optimization via: Architecture + Examples + Token Allocation
         
├─ combined_optimized
│  Architecture: Hybrid Adaptive Reasoning
│  Examples: 9 total (3 per group: Mexican, LGBTQ+, Middle Eastern)
│  Tokens: 650
│  Reasoning: Confidence-based (high→direct, low→multi-perspective)
│  
├─ combined_focused  
│  Architecture: Direct Binary + Cultural Context
│  Examples: 9 total (3 per group, priority-ordered)
│  Tokens: 500
│  Reasoning: Cultural awareness framework (4 dimensions)
│
├─ combined_conservative
│  Architecture: Minimal Overhead
│  Examples: 6 total (2 per group, priority-ordered)
│  Tokens: 400
│  Reasoning: Streamlined classification

All configurations: temp=1.0 (fixed), response_format=json_object
Priority Ordering: Mexican/Latino FIRST (immigration-based hate detection)
```

### Strategy Taxonomy

**Combined Optimized (Hybrid Adaptive Reasoning)**: Implements confidence-based multi-stage analysis combining adaptive complexity with nine few-shot examples (three per demographic group). High-confidence cases trigger direct classification for efficiency, while ambiguous cases invoke multi-perspective analysis across policy violation, community impact, cultural context, and language patterns. The 650-token allocation accommodates confidence assessment and extended reasoning chains. Explicit LGBTQ+ in-group reclamation guidance distinguishes community self-identification from out-group attacks. Mexican/Latino examples prioritize immigration-based hate detection (generalizations, dehumanization, coded language) versus policy discussion. Outputs include classification, confidence level (high/medium/low), and rationale.

**Combined Focused (Direct Binary + Cultural Context)**: Streamlines classification through cultural awareness framework integrating four analytical dimensions: historical discrimination patterns, power dynamics, community norms, and intent versus impact assessment. Nine priority-ordered examples (Mexican/Latino first) provide compact pattern encoding without verbose explanations. The 500-token budget balances cultural context depth with computational efficiency. Explicit recall priority guidance addresses subtle and coded hate detection, with precision guards preventing over-flagging of factual Middle Eastern content. LGBTQ+ harm-versus-affirmation framing distinguishes out-group attacks from in-group reclamation.

**Combined Conservative (Minimal Overhead)**: Prioritizes efficiency through six essential examples (two per group) with streamlined three-question evaluation framework. The 400-token constraint enforces concise reasoning suitable for high-throughput scenarios. Priority ordering maintains Mexican/Latino examples first despite reduced example count. Precision guards prevent Middle Eastern factual content over-flagging. Detection emphasis explicitly clarifies that subtle or coded hate constitutes hate speech without requiring explicit slurs, addressing under-detection risks in efficiency-focused configurations.

### Experimental Rationale

The Combined GPT-5 framework operationalizes architectural optimization as the primary performance lever when hyperparameter tuning is constrained. The hybrid adaptive reasoning architecture (combined_optimized) tests whether confidence-based complexity adjustment improves accuracy by allocating reasoning resources proportional to case difficulty. The direct binary plus cultural context architecture (combined_focused) validates whether integrated fairness frameworks improve demographic bias metrics without multi-stage reasoning overhead. The minimal overhead architecture (combined_conservative) establishes efficiency boundaries, quantifying performance-speed trade-offs.

Few-shot example integration addresses documented bias patterns: Mexican/Latino immigration-based hate under-detection, LGBTQ+ in-group reclamation over-flagging, and Middle Eastern terrorism generalization failures. Priority ordering (Mexican examples first) leverages recency bias in context windows to mitigate the most severe fairness issues. Token allocation strategy reflects empirical findings that GPT-5 performance plateaus beyond 600 tokens, with combined_optimized's 650-token budget targeting optimal balance while combined_conservative's 400-token constraint tests minimum viable configuration. The framework validates whether architectural prompt engineering achieves performance gains comparable to GPT-OSS combined's hyperparameter-driven optimization despite API constraints.

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
