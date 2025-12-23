# In-Context Fine-Tuning (IFT) Approach Implementation

## Introduction

This section presents the implementation architecture of the In-Context Fine-Tuning (IFT) approach for hate speech detection, wherein prompt engineering strategies are systematically evaluated without parameter updates to the base language model. The implementation encompasses a modular pipeline architecture, JSON-based template configuration system, and comprehensive evaluation framework designed to enable reproducible experimentation across multiple prompt strategies.

## Pipeline Architecture

The IFT validation framework implements a modular, package-based architecture designed to support concurrent strategy evaluation, flexible model configuration, and comprehensive performance analytics. The system architecture comprises four primary components: (1) the orchestration layer, (2) the connector subsystem, (3) the data loading subsystem, and (4) the metrics computation subsystem.

### Orchestration Layer

The orchestration layer, implemented in `prompt_runner.py`, serves as the primary command-line interface and coordinates all pipeline operations. This component implements the following core functionalities:

1. **Strategy Management**: The orchestrator loads and manages multiple prompt strategies from JSON configuration files, enabling flexible strategy selection and concurrent evaluation across diverse prompt formulations.

2. **Concurrent Execution**: The system implements a thread-pool based concurrent execution model with configurable worker pools (default: 5 workers) and batch processing (default: 10 samples per batch). This design enables efficient throughput while respecting API rate limits.

3. **Incremental Storage**: Results are persisted incrementally to CSV format during validation, preventing memory overflow during large-scale experiments. Each validation sample is written immediately upon completion, ensuring data persistence in the event of execution interruption.

4. **Rate Limiting and Retry Logic**: The orchestrator implements intelligent retry mechanisms with exponential backoff (delay = min(30, 2^attempt + random(0,1))) and rate limit detection, parsing HTTP response headers to monitor remaining request quotas and token budgets.

The orchestration workflow follows a five-stage pipeline:

```
Stage 1: Configuration Loading
    ├─ Model configuration from YAML
    ├─ Strategy templates from JSON
    └─ Dataset sampling and preparation

Stage 2: Connection Validation
    ├─ Azure AI endpoint connectivity test
    └─ Authentication credential verification

Stage 3: Concurrent Strategy Execution
    ├─ Thread pool initialization (n=max_workers)
    ├─ Batch processing (k=batch_size samples)
    ├─ Strategy application with retry logic
    └─ Incremental result persistence

Stage 4: Metrics Computation
    ├─ Performance metrics (accuracy, F1, precision, recall)
    ├─ Confusion matrix generation
    └─ Bias metrics by target group

Stage 5: Result Aggregation and Reporting
    ├─ Run-specific directory creation (run_YYYYMMDD_HHMMSS/)
    ├─ Multi-file output generation
    └─ Human-readable evaluation report
```

### Connector Subsystem

The connector subsystem (`connector/azureai_connector.py`) provides an abstraction layer over the Azure AI Inference SDK, enabling multi-model support through YAML-based configuration. Key architectural features include:

1. **Configuration Management**: The `ModelConfigLoader` class implements YAML parsing with environment variable substitution (${VAR_NAME} syntax), enabling secure credential management and deployment flexibility across development, staging, and production environments.

2. **Multi-Model Support**: The connector supports arbitrary model configurations through a unified interface, abstracting provider-specific details while maintaining parameter flexibility (temperature, max_tokens, top_p, etc.).

3. **Message Handling**: The system implements the OpenAI message format with `SystemMessage` and `UserMessage` objects, enabling consistent prompt construction across different model providers.

The connection workflow implements defensive programming practices:

```python
# Pseudo-code representation of connection logic
def initialize_client(endpoint, api_key, model_deployment):
    validate_credentials(endpoint, api_key)
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    verify_model_availability(client, model_deployment)
    return client

def complete_with_retry(messages, parameters, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            response = client.complete(messages, **parameters)
            parse_rate_limit_headers(response)
            return response
        except RateLimitException:
            delay = exponential_backoff(attempt)
            time.sleep(delay)
        except NetworkException:
            if attempt == max_retries:
                raise
    raise MaxRetriesExceeded
```

### Data Loading Subsystem

The data loading subsystem (`loaders/`) implements flexible dataset management with support for both large-scale unified datasets and curated canned test sets. The architecture comprises two primary components:

1. **UnifiedDatasetLoader**: Implements stratified sampling with configurable sample sizes, ensuring balanced representation across target groups (LGBTQ+, Middle Eastern, Mexican/Latino) and binary labels (hate/normal).

2. **StrategyTemplatesLoader**: Parses JSON-based strategy configurations and instantiates `PromptStrategy` objects with associated `PromptTemplate` components. This separation enables independent evolution of dataset formats and prompt templates.

The loader implements the following data structure:

```python
@dataclass
class PromptTemplate:
    system_prompt: str      # Model context and role definition
    user_template: str      # Input template with {text} placeholder

@dataclass
class PromptStrategy:
    name: str               # Strategy identifier (e.g., 'baseline')
    description: str        # Human-readable strategy explanation
    template: PromptTemplate  # Prompt template components
    parameters: Dict[str, Any]  # Model-specific parameters
```

### Metrics Computation Subsystem

The metrics subsystem (`metrics/evaluation_metrics_calc.py`) implements comprehensive evaluation analytics using scikit-learn metrics. The system computes:

1. **Classification Metrics**: Accuracy, precision, recall, and F1-score using binary classification over hate/normal labels.

2. **Confusion Matrix**: True positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for detailed error analysis.

3. **Bias Metrics**: False positive rates (FPR = FP/(FP+TN)) and false negative rates (FNR = FN/(FN+TP)) stratified by target group, enabling fairness evaluation across protected demographics.

The metrics are computed from stored CSV results rather than in-memory data structures, enabling post-hoc analysis and reproducible metric calculation:

```python
@dataclass
class PerformanceMetrics:
    strategy: str
    accuracy: float          # (TP + TN) / (TP + TN + FP + FN)
    precision: float         # TP / (TP + FP)
    recall: float            # TP / (TP + FN)
    f1_score: float          # 2 * (precision * recall) / (precision + recall)
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

@dataclass
class BiasMetrics:
    persona_tag: str         # Target group identifier
    sample_count: int
    false_positive_rate: float  # Rate of misclassifying benign as hate
    false_negative_rate: float  # Rate of missing actual hate speech
    # ... confusion matrix components
```

## Template JSON Configuration

The IFT framework implements a declarative JSON-based configuration system for prompt strategies, enabling rapid experimentation without code modification. Each strategy configuration encapsulates the complete prompt engineering design, including system-level context, user input templates, and model-specific parameters.

### JSON Schema Structure

The strategy configuration follows a hierarchical JSON schema:

```json
{
  "strategies": {
    "<strategy_name>": {
      "name": "<strategy_identifier>",
      "description": "<human_readable_description>",
      "system_prompt": "<system_level_context>",
      "user_template": "<user_input_template_with_placeholders>",
      "parameters": {
        "max_tokens": <integer>,
        "temperature": <float_0_to_2>,
        "top_p": <float_0_to_1>,
        "frequency_penalty": <float_-2_to_2>,
        "presence_penalty": <float_-2_to_2>,
        "response_format": "<text|json_object>"
      }
    }
  }
}
```

Key schema components:

1. **Strategy Name**: Unique identifier used for command-line selection and result tracking (e.g., `baseline`, `combined_v5_implicit_examples`).

2. **Description**: Metadata explaining the strategy's theoretical foundation and expected behavior, enabling documentation and reproducibility.

3. **System Prompt**: The system-level message that establishes the model's role, output format constraints, and high-level classification rules. This component remains constant across all input samples for a given strategy.

4. **User Template**: The per-sample input template containing placeholder variables (e.g., `{text}`, `{target_group}`) that are instantiated with actual sample data during execution.

5. **Parameters**: Model-specific generation parameters controlling output diversity (temperature, top_p), repetition (frequency_penalty, presence_penalty), length constraints (max_tokens), and structured output formatting (response_format).

### Parameter Semantics

The generation parameters implement standard large language model control mechanisms:

- **max_tokens** (default: 512): Maximum output length, controlling computational cost and ensuring concise rationales.

- **temperature** (default: 0.1): Sampling temperature where lower values (0.0-0.5) produce deterministic outputs suitable for classification tasks, while higher values (0.5-2.0) increase output diversity.

- **top_p** (default: 1.0): Nucleus sampling threshold; values below 1.0 restrict sampling to the top cumulative probability mass.

- **frequency_penalty** (default: 0.0): Penalizes token repetition based on frequency in the generated text (-2.0 to 2.0).

- **presence_penalty** (default: 0.0): Penalizes token repetition based on presence in the generated text (-2.0 to 2.0).

- **response_format** (default: "json_object"): Enforces structured JSON output, enabling robust parsing and eliminating free-form text variability.

### Template Variable Substitution

The system implements runtime template variable substitution using string formatting:

```python
def format_prompt(user_template: str, text: str, target_group: str) -> str:
    """
    Instantiate user template with sample-specific values.
    
    Args:
        user_template: Template string with {text} and {target_group} placeholders
        text: Input text to classify
        target_group: Target demographic group (e.g., 'lgbtq', 'mexican', 'arab')
    
    Returns:
        Formatted prompt ready for model input
    """
    return user_template.format(text=text, target_group=target_group)
```

This design enables (1) separation of prompt logic from runtime data, (2) consistent formatting across samples, and (3) easy template modification without code changes.

## Sample Prompt Templates

This section presents two representative prompt templates that illustrate the fundamental approaches in the IFT framework: the baseline strategy and the combined strategy.

### Baseline Strategy

The baseline strategy implements minimal context, establishing a reference performance level without policy guidance or community perspectives:

**System Prompt:**
```
You are a content moderation assistant trained to classify social media posts 
as either 'hate' or 'normal'.

You MUST respond in strict JSON format with the following structure:
{"classification": "hate", "rationale": "brief explanation"} or 
{"classification": "normal", "rationale": "brief explanation"}

Classification Rules:
- Use EXACTLY 'hate' if the text contains hate speech
- Use EXACTLY 'normal' if the text does not contain hate speech
- NO other labels are allowed (not 'not_hate', 'hateful', etc.)
- Always provide a brief rationale (1-2 sentences) explaining your decision

Base your decision on general understanding of hateful language and social norms.
```

**User Template:**
```
Classify the following text as either 'hate' or 'normal'.

Text: "{text}"

Respond in JSON format: {{"classification": "[hate/normal]", 
                         "rationale": "[brief explanation]"}}
```

**Parameters:**
- max_tokens: 512
- temperature: 0.1
- top_p: 1.0
- response_format: json_object

**Strategy Characteristics:**
- Word count: ~120 words
- No policy guidance or community perspectives
- Relies on model's pre-training for hate speech understanding
- Minimal prompt engineering intervention

### Combined Strategy (Policy + Persona)

The combined strategy integrates platform policy guidelines with community perspectives, representing a comprehensive prompt engineering approach:

**System Prompt:**
```
You are a content moderation assistant evaluating social media posts for hate 
speech using both X platform's official hateful conduct policy and multiple 
community perspectives from the most represented groups in the dataset.

You MUST respond in strict JSON format with the following structure:
{"classification": "hate", "rationale": "brief explanation"} or 
{"classification": "normal", "rationale": "brief explanation"}

Classification Rules:
- Use EXACTLY 'hate' if the text violates X policy OR is harmful from any 
  community perspective
- Use EXACTLY 'normal' if the text does not violate X policy AND is not harmful 
  from any community perspective
- NO other labels are allowed
- Always provide a brief rationale (1-2 sentences) explaining your decision
- Focus on detecting slurs, dehumanization, stereotypes, and incitement targeting 
  protected groups
- Pay special attention to implicit hate and coded language
```

**User Template:**
```
X PLATFORM HATEFUL CONDUCT POLICY: Content violates policy if it directly attacks 
people based on race, ethnicity, national origin, sexual orientation, gender, 
gender identity, religious affiliation, age, disability, or serious disease. 
Specifically prohibited:
- Slurs and derogatory language (e.g., 'faggots', 'beaners', targeting identity)
- Dehumanization and derogatory characterizations
- Incitement of fear, harassment, or discrimination
- Reinforcing harmful stereotypes about protected groups

Analyze this text using both X policy and community perspectives:

LGBTQ+ Perspective: Does this attack sexual orientation, gender identity, or use 
anti-LGBTQ+ slurs/stereotypes?
Middle Eastern Perspective: Does this target Arabs, Muslims, or Middle Eastern 
people with slurs, stereotypes, or discrimination?
Mexican/Latino Perspective: Does this use anti-Mexican/Latino slurs (like 'beaners') 
or promote harmful stereotypes?

Text: "{text}"

Provide a consolidated policy-based decision as EXACTLY one of these labels:
- hate (if text violates X policy OR is harmful from any community perspective)
- normal (if text does not violate X policy AND is not harmful from any community 
  perspective)

Respond in JSON format: {{"classification": "[hate/normal]", 
                         "rationale": "[brief explanation]"}}
```

**Parameters:**
- max_tokens: 768
- temperature: 0.2
- top_p: 0.9
- frequency_penalty: 0.1
- presence_penalty: 0.1
- response_format: json_object

**Strategy Characteristics:**
- Word count: ~280 words
- Explicit policy enumeration with example slurs
- Three community perspectives (LGBTQ+, Middle Eastern, Mexican/Latino)
- Combines policy-based and community-based evaluation frameworks

**Design Rationale:**
This strategy implements a multi-faceted evaluation framework that combines two complementary approaches: (1) explicit platform policy guidelines derived from X (formerly Twitter) hateful conduct policy, and (2) community-based perspective-taking from the three most represented demographic groups in the dataset. The policy component provides concrete examples of prohibited content (slurs, dehumanization, incitement), while the community perspectives operationalize the harm assessment through targeted questions that surface group-specific vulnerabilities. This dual-framework design aims to capture both rule-based violations and context-dependent harm that may vary across protected groups.

## Final Prompts: Construction and Execution

The final prompts submitted to the language model are constructed through a multi-stage assembly process that combines static template components with runtime sample data.

### Message Construction Pipeline

The system implements the standard OpenAI message format with separate system and user roles:

```python
def construct_messages(strategy: PromptStrategy, sample: Dict[str, Any]) -> List[Message]:
    """
    Construct message list for model inference.
    
    Args:
        strategy: PromptStrategy object containing template and parameters
        sample: Dictionary with 'text' and 'target_group_norm' fields
    
    Returns:
        List of Message objects (SystemMessage, UserMessage)
    """
    # Extract template components
    system_prompt = strategy.template.system_prompt
    user_template = strategy.template.user_template
    
    # Format user template with sample data
    user_prompt = user_template.format(
        text=sample['text'],
        target_group=sample.get('target_group_norm', 'general')
    )
    
    # Construct message list
    messages = [
        SystemMessage(content=system_prompt),
        UserMessage(content=user_prompt)
    ]
    
    return messages
```

### Example Final Prompt: Baseline Strategy

For the sample text "Those faggots make me sick" (label: hate, target_group: lgbtq), the baseline strategy produces:

**SystemMessage:**
```
You are a content moderation assistant trained to classify social media posts 
as either 'hate' or 'normal'.

You MUST respond in strict JSON format with the following structure:
{"classification": "hate", "rationale": "brief explanation"} or 
{"classification": "normal", "rationale": "brief explanation"}

Classification Rules:
- Use EXACTLY 'hate' if the text contains hate speech
- Use EXACTLY 'normal' if the text does not contain hate speech
- NO other labels are allowed (not 'not_hate', 'hateful', etc.)
- Always provide a brief rationale (1-2 sentences) explaining your decision

Base your decision on general understanding of hateful language and social norms.
```

**UserMessage:**
```
Classify the following text as either 'hate' or 'normal'.

Text: "Those faggots make me sick"

Respond in JSON format: {"classification": "[hate/normal]", 
                         "rationale": "[brief explanation]"}
```

**Expected Response:**
```json
{
  "classification": "hate",
  "rationale": "The text contains an anti-LGBTQ+ slur ('faggots') used in a 
               derogatory context to express disgust, constituting hate speech 
               targeting sexual orientation."
}
```

### Example Final Prompt: Combined Strategy

For the same sample text, the combined strategy produces:

**SystemMessage:**
```
You are a content moderation assistant evaluating social media posts for hate 
speech using both X platform's official hateful conduct policy and multiple 
community perspectives from the most represented groups in the dataset.

You MUST respond in strict JSON format with the following structure:
{"classification": "hate", "rationale": "brief explanation"} or 
{"classification": "normal", "rationale": "brief explanation"}

Classification Rules:
- Use EXACTLY 'hate' if the text violates X policy OR is harmful from any 
  community perspective
- Use EXACTLY 'normal' if the text does not violate X policy AND is not harmful 
  from any community perspective
- NO other labels are allowed
- Always provide a brief rationale (1-2 sentences) explaining your decision
- Focus on detecting slurs, dehumanization, stereotypes, and incitement targeting 
  protected groups
- Pay special attention to implicit hate and coded language
```

**UserMessage:**
```
X PLATFORM HATEFUL CONDUCT POLICY: Content violates policy if it directly attacks 
people based on race, ethnicity, national origin, sexual orientation, gender, 
gender identity, religious affiliation, age, disability, or serious disease. 
Specifically prohibited:
- Slurs and derogatory language (e.g., 'faggots', 'beaners', targeting identity)
- Dehumanization and derogatory characterizations
- Incitement of fear, harassment, or discrimination
- Reinforcing harmful stereotypes about protected groups

Analyze this text using both X policy and community perspectives:

LGBTQ+ Perspective: Does this attack sexual orientation, gender identity, or use 
anti-LGBTQ+ slurs/stereotypes?
Middle Eastern Perspective: Does this target Arabs, Muslims, or Middle Eastern 
people with slurs, stereotypes, or discrimination?
Mexican/Latino Perspective: Does this use anti-Mexican/Latino slurs (like 'beaners') 
or promote harmful stereotypes?

Text: "Those faggots make me sick"

Provide a consolidated policy-based decision as EXACTLY one of these labels:
- hate (if text violates X policy OR is harmful from any community perspective)
- normal (if text does not violate X policy AND is not harmful from any community 
  perspective)

Respond in JSON format: {"classification": "[hate/normal]", 
                         "rationale": "[brief explanation]"}
```

**Expected Response:**
```json
{
  "classification": "hate",
  "rationale": "Text violates X platform policy by using anti-LGBTQ+ slur 
               ('faggots') in derogatory context expressing disgust toward 
               sexual orientation. LGBTQ+ perspective confirms this constitutes 
               targeted attack on protected characteristic."
}
```

**Comparative Analysis:**
The combined strategy provides more contextual rationale by explicitly referencing the policy framework and community perspective used in the evaluation. While the baseline strategy correctly identifies the hate speech based on general understanding, the combined strategy's response demonstrates application of specific policy criteria (slur identification, protected characteristic targeting) and community harm assessment. However, this enhanced interpretability comes at the cost of significantly increased prompt length (~280 words vs. ~120 words), which may introduce noise and cognitive load for the model.

### Response Parsing and Label Standardization

The system implements robust response parsing with fallback mechanisms:

```python
def parse_response(response_text: str) -> Tuple[str, str]:
    """
    Parse model response and extract classification + rationale.
    
    Args:
        response_text: Raw model response (JSON or text)
    
    Returns:
        Tuple of (predicted_label, rationale)
    """
    try:
        # Attempt JSON parsing
        data = json.loads(response_text)
        raw_label = data.get('classification', '').lower()
        rationale = data.get('rationale', 'No rationale provided')
        
    except json.JSONDecodeError:
        # Fallback: Text parsing for classification keywords
        raw_label = extract_label_from_text(response_text)
        rationale = response_text
    
    # Standardize label variants to binary labels
    predicted_label = standardize_label(raw_label)
    
    return predicted_label, rationale

def standardize_label(raw_label: str) -> str:
    """
    Map label variations to standardized binary labels.
    
    Mappings:
    - 'hate', 'hateful', 'hate speech' → 'hate'
    - 'normal', 'not hate', 'not_hate', 'benign' → 'normal'
    """
    hate_variants = {'hate', 'hateful', 'hate speech', 'hate_speech'}
    normal_variants = {'normal', 'not hate', 'not_hate', 'benign', 'acceptable'}
    
    if raw_label in hate_variants:
        return 'hate'
    elif raw_label in normal_variants:
        return 'normal'
    else:
        logger.warning(f"Unknown label '{raw_label}', defaulting to 'normal'")
        return 'normal'
```

This two-stage parsing ensures robustness against (1) JSON formatting errors, (2) label vocabulary variations, and (3) model non-compliance with output format specifications.

### Execution Workflow

The complete execution workflow for a single sample proceeds as follows:

```
1. Sample Selection
   └─ Load sample from dataset (text, label_binary, target_group_norm)

2. Strategy Application
   ├─ Load strategy template from JSON configuration
   ├─ Format user template with sample.text
   └─ Construct [SystemMessage, UserMessage] list

3. Model Inference
   ├─ Retrieve model parameters from strategy configuration
   ├─ Submit messages to Azure AI endpoint via connector
   ├─ Implement retry logic with exponential backoff
   └─ Parse rate limit headers for throttling detection

4. Response Processing
   ├─ Parse JSON response (with text fallback)
   ├─ Standardize label to binary format
   ├─ Extract rationale for interpretability
   └─ Record response time for performance analysis

5. Result Persistence
   ├─ Write to validation_results.csv incrementally
   ├─ Record: strategy, sample_id, text, true_label, predicted_label, 
   │          response_time, rationale
   └─ Flush to disk immediately (prevent data loss)

6. Metrics Computation (post-validation)
   ├─ Load validation_results.csv from run directory
   ├─ Compute classification metrics (accuracy, F1, precision, recall)
   ├─ Generate confusion matrix (TP, TN, FP, FN)
   ├─ Calculate bias metrics by target group (FPR, FNR)
   └─ Write performance_metrics.csv and evaluation_report.txt
```

### Reproducibility Considerations

The implementation ensures experimental reproducibility through:

1. **Configuration Versioning**: All strategy templates, model parameters, and dataset selections are serialized in output directories with timestamp-based run IDs (e.g., `run_20251102_191102/`).

2. **Random Seed Control**: Dataset sampling implements configurable random seeds (default: 42), enabling deterministic sample selection across runs.

3. **Metadata Logging**: Each run directory contains complete execution metadata including command-line arguments, prompt template file paths, model configuration, and dataset source.

4. **Result Immutability**: Validation results are write-once files; metrics recalculation uses the `--metrics-only` flag to recompute from stored results without re-running inference.

This architecture enables rigorous comparison across prompt strategies, model configurations, and dataset variations while maintaining experimental integrity and audit trail completeness.

## Summary

The IFT approach implementation demonstrates a production-ready architecture for systematic prompt engineering evaluation. The modular design separates concerns across orchestration, model connection, data loading, and metrics computation, enabling independent evolution of each subsystem. The JSON-based configuration system provides declarative strategy specification without code modification, supporting rapid experimentation across the prompt engineering design space.

The sample prompt templates illustrate the progression from baseline minimal prompts (~120 words) to comprehensive combined strategies (~280 words). The final prompt construction pipeline implements robust message assembly, response parsing, and label standardization, ensuring reliable classification output across model variations.

This implementation serves as the experimental foundation for the five-iteration prompt engineering study, enabling systematic evaluation of verbosity versus compression hypotheses. The two exemplar strategies presented here (baseline and combined) illustrate the fundamental tension between minimal intervention and comprehensive guidance, with empirical results demonstrating that strategic prompt design can yield measurable improvements in classification performance while maintaining computational efficiency.
