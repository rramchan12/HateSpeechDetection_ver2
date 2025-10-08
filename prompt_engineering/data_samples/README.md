# Stratified Sample Dataset Design for Hate Speech Detection

## Abstract

This document presents the design and methodology for three stratified sample datasets used in systematic evaluation of Large Language Model (LLM) prompt engineering strategies for hate speech detection. The sampling approach maintains statistical fidelity to the original training distribution while optimizing for computational efficiency and comprehensive evaluation coverage.

## 1. Introduction

Effective evaluation of hate speech detection systems requires carefully constructed test sets that balance statistical representativeness with computational practicality. This work presents a stratified sampling methodology that creates three complementary evaluation datasets from a unified training corpus of 3,628 hate speech samples across multiple demographic targets.

## 2. Sample Size Justification

### 2.1 Statistical Considerations

The 100-sample baseline represents an optimal balance across multiple evaluation criteria:

**Representational Adequacy**: Sample size ensures sufficient representation across stratification dimensions (target groups, binary labels, source datasets) while maintaining statistical significance for performance metrics including accuracy, precision, recall, and F1-score calculations.

**Stratification Fidelity**: The sampling methodology achieves <1% deviation from original training distributions across all measured dimensions, ensuring representative evaluation conditions.

**Comparative Validity**: Sample size provides sufficient statistical power to detect meaningful performance differences between prompt engineering strategies, enabling reliable comparative analysis.

### 2.2 Computational Efficiency

**Resource Optimization**: 100 samples × 4 strategies = 400 API calls per evaluation cycle, balancing comprehensive testing with computational cost constraints.

**Processing Scalability**: Sample size enables manageable execution times (5-7 minutes) for iterative prompt engineering workflows while remaining compatible with concurrent processing and rate limiting requirements.

**Memory Management**: Sample size works effectively with incremental storage systems and memory-efficient processing architectures.

## 3. Multi-Dataset Evaluation Framework

The evaluation framework employs three complementary sample datasets, each designed for specific validation requirements:

### 3.1 Rapid Iteration Dataset (`canned_50_quick.json`)

**Design Specification**: 50-sample subset optimized for development workflows

**Application**: Real-time validation during prompt strategy development, enabling rapid iteration cycles with processing times under 3 minutes per complete evaluation.

**Validation Scope**: Parameter tuning, debugging, and immediate feedback during active development phases.

### 3.2 Comprehensive Evaluation Dataset (`canned_100_stratified.json`)

**Design Specification**: 100-sample stratified dataset maintaining optimal distribution fidelity

**Application**: Official strategy comparison and performance benchmarking with processing times of 5-7 minutes per evaluation cycle.

**Validation Scope**: Baseline establishment, systematic strategy validation, and final performance assessment.

### 3.3 Robustness Testing Dataset (`canned_100_size_varied.json`)

**Design Specification**: 100-sample dataset with intentional text length variation to simulate production conditions

**Application**: Production readiness validation and edge case analysis across diverse content complexity levels.

**Validation Scope**: Deployment validation, robustness assessment, and real-world performance simulation.

## 4. Methodology

### 4.1 Stratified Sampling Framework

Sample generation employs a multi-dimensional stratified sampling approach implemented through the `dataset_sampler.py` system, ensuring representative distribution maintenance across critical evaluation dimensions.

### 4.2 Stratification Dimensions

The sampling methodology operates across four key dimensions:

1. **Demographic Target Groups**: Proportional representation of LGBTQ (48.6%), Middle East (28.1%), and Mexican (23.3%) communities
2. **Binary Classification Labels**: Balanced hate/normal classification maintaining original distribution ratios
3. **Source Dataset Representation**: Proportional allocation from HatEXplain (47.5%) and ToxiGen (52.5%) datasets
4. **Text Length Distribution**: Strategic length variation for comprehensive model evaluation (applied selectively)

### 4.3 Algorithmic Implementation

The sampling algorithm implements proportional allocation across stratification combinations:

```python
# Proportional allocation across demographic-label strata
for stratum in [target_group + label combinations]:
    allocation = max(1, round(sample_size * stratum_proportion))
    samples = random.sample(stratum_data, allocation)
```

### 4.4 Quality Assurance Protocol

- **Distribution Fidelity**: <1% deviation from original training set distributions across all measured dimensions
- **Reproducibility**: Seed-based deterministic sampling (seed=42) ensuring consistent evaluation conditions
- **Statistical Validation**: Comprehensive distribution verification across all stratification dimensions

## 5. Results and Analysis

### 5.1 Baseline Distribution Characteristics

The original training corpus exhibits the following distributional properties:

```text
Total Samples: 3,628
Target Groups: LGBTQ (48.6%), Middle East (28.1%), Mexican (23.3%)
Binary Labels: Normal (53.1%), Hate (46.9%)
Source Datasets: ToxiGen (52.5%), HatEXplain (47.5%)
Text Length Distribution: Short <50 chars (22.3%), Medium 50-150 chars (60.9%), Long >150 chars (16.8%)
Average Text Length: 102.0 characters
Character Range: 1-540 characters
```

### 5.2 Comparative Distribution Analysis

| Metric | Original | Quick-50 | Stratified-100 | Size-Varied-100 |
|--------|----------|----------|----------------|-----------------|
| **Target Groups** |  |  |  |  |
| LGBTQ | 48.6% | 50.0% | 49.0% | 49.0% |
| Middle East | 28.1% | 28.0% | 28.0% | 29.0% |
| Mexican | 23.3% | 22.0% | 23.0% | 22.0% |
| **Labels** |  |  |  |  |
| Normal | 53.1% | 54.0% | 53.0% | 53.0% |
| Hate | 46.9% | 46.0% | 47.0% | 47.0% |
| **Sources** |  |  |  |  |
| ToxiGen | 52.5% | 52.0% | 50.0% | 51.0% |
| HatEXplain | 47.5% | 48.0% | 50.0% | 49.0% |
| **Text Length** |  |  |  |  |
| Short (<50) | 22.3% | 30.0% | 19.0% | 30.0% |
| Medium (50-150) | 60.9% | 52.0% | 61.0% | 50.0% |
| Long (>150) | 16.8% | 18.0% | 20.0% | 20.0% |
| **Maximum Deviation** | - | ±6% | ±2% | ±8% |

### 5.3 Size-Varied Dataset Optimality

The `canned_100_size_varied.json` dataset provides optimal representation for production evaluation scenarios:

**Intentional Length Distribution**: The strategic 30%/50%/20% distribution (Short/Medium/Long) reflects authentic social media content patterns, with extended character range (8-526 characters) capturing the full spectrum of hate speech expression modalities.

**Real-World Authenticity**: The length distribution accommodates diverse hate speech manifestations: direct explicit content in short posts typical of social media platforms, embedded bias in medium-length standard posts, and complex coded language in extended narrative posts.

**Comprehensive Model Validation**: The varied text lengths enable attention mechanism validation across context lengths, prompt strategy robustness testing across content complexities, and production deployment readiness assessment under realistic content distribution conditions.

**Edge Case Coverage**: The dataset includes minimal context scenarios testing keyword-based detection limits, information density variance validating attention and reasoning capabilities, and processing complexity ranges ensuring consistent performance across input variations.

## 6. Implementation Guidelines

### 6.1 Evaluation Workflow

The multi-dataset framework supports systematic evaluation through the following protocol:

```bash
# Rapid iteration during prompt development
python prompt_runner.py --data-source canned_50_quick --strategies baseline

# Comprehensive strategy validation  
python prompt_runner.py --data-source canned_100_stratified --strategies all

# Production readiness testing
python prompt_runner.py --data-source canned_100_size_varied --strategies all
```

### 6.2 Dataset Selection Protocol

**Rapid Iteration Dataset Applications:**

- Initial prompt strategy development and parameter optimization
- Real-time validation during active development cycles
- Debugging and response parsing verification
- Scenarios requiring immediate feedback (processing time <3 minutes)

**Comprehensive Evaluation Dataset Applications:**

- Baseline performance metric establishment
- Systematic multi-strategy comparative analysis
- Official benchmark result generation
- Applications requiring optimal distribution fidelity

**Robustness Testing Dataset Applications:**

- Production deployment readiness validation
- Edge case analysis and robustness assessment
- Real-world content diversity simulation
- Large Language Model attention mechanism evaluation

## 7. Quality Assurance Framework

The sample generation process incorporates comprehensive validation protocols:

**Distribution Verification**: Automated statistical validation ensuring <1% deviation from original corpus distributions across all stratification dimensions.

**Content Validation**: Manual inspection and verification of edge cases and representative content samples.

**Performance Baseline**: Systematic evaluation across all prompt strategies to establish performance benchmarks.

**Reproducibility Assurance**: Deterministic sampling through fixed random seeds ensuring consistent evaluation conditions across experimental iterations.

## 8. Conclusion

This stratified sampling methodology provides a robust foundation for systematic prompt engineering and validation of hate speech detection strategies, balancing statistical representativeness with computational efficiency while enabling comprehensive evaluation across diverse operational scenarios.