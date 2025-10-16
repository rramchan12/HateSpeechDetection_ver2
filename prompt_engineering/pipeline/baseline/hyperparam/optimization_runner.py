#!/usr/bin/env python3
"""
Optimization Runner - Comprehensive Hyperparameter Optimization Demo

This runner demonstrates the full capabilities of the HyperparameterOptimiser class
with multiple scoring strategies and comprehensive analysis.
"""

import argparse
import sys
from pathlib import Path

# Add the hyperparam directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from hyperparameter_optimiser import HyperparameterOptimiser


def extract_run_info(run_id):
    """Extract run information from run_id by reading the evaluation report."""
    import re
    
    # Search for the run_id directory anywhere in the outputs folder structure
    outputs_base_paths = [
        '../../../../outputs',
        '../../../outputs'
    ]
    
    evaluation_report_path = None
    run_path = None
    
    # Search recursively for the run_id directory
    for outputs_path in outputs_base_paths:
        outputs_dir = Path(outputs_path)
        if outputs_dir.exists():
            # Search for run directory anywhere in outputs
            for run_dir in outputs_dir.rglob(run_id):
                if run_dir.is_dir():
                    # Look for evaluation report file in this directory
                    timestamp = run_id.replace("run_", "")
                    eval_report = run_dir / f'evaluation_report_{timestamp}.txt'
                    if eval_report.exists():
                        evaluation_report_path = eval_report
                        run_path = str(run_dir)
                        break
            if evaluation_report_path:
                break
    
    if not evaluation_report_path or not evaluation_report_path.exists():
        # Fallback to default configuration if report not found
        print(f"Warning: Could not find evaluation report for {run_id}, using defaults")
        return {
            'run_path': f'../../../../outputs/baseline_v1/{run_id}',
            'template_file': 'baseline_v1.json',
            'model_name': 'gpt-oss-120b',
            'prompt_templates_dir': '../../../prompt_templates'
        }
    
    # Parse the evaluation report to extract information
    try:
        with open(evaluation_report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract model name
        model_match = re.search(r'Model:\s*(.+)', content)
        model_name = model_match.group(1).strip() if model_match else 'unknown'
        
        # Extract prompt template file
        template_match = re.search(r'Prompt Template File:\s*(.+)', content)
        template_file = template_match.group(1).strip() if template_match else 'unknown.json'
        
        # Extract data source
        data_source_match = re.search(r'Data Source:\s*(.+)', content)
        data_source = data_source_match.group(1).strip() if data_source_match else 'unknown'
        
        return {
            'run_path': run_path,
            'template_file': template_file,
            'model_name': model_name,
            'data_source': data_source,
            'prompt_templates_dir': '../../../prompt_templates'
        }
        
    except Exception as e:
        print(f"Error parsing evaluation report {evaluation_report_path}: {e}")
        # Fallback to default configuration
        return {
            'run_path': f'../../../../outputs/baseline_v1/{run_id}',
            'template_file': 'baseline_v1.json',
            'model_name': 'gpt-oss-120b',
            'prompt_templates_dir': '../../../prompt_templates'
        }


def run_performance_focused_analysis(run_infos, output_dir):
    """Run performance-focused analysis (90% performance, 10% bias)."""
    print('\nTEST 1: Performance-Focused Analysis (90% performance, 10% bias)')
    
    optimizer = HyperparameterOptimiser(
        prompt_templates_dir=run_infos[0]['prompt_templates_dir'],  # Use first run's template dir
        output_dir=output_dir,
        bias_weight=0.1,
        performance_weight=0.9
    )

    run_configs = []
    for run_info in run_infos:
        run_configs.append({
            'run_path': run_info['run_path'],
            'template_file': run_info['template_file'],
            'model_name': run_info['model_name']
        })

    optimizer.analyze_runs(run_configs)
    optimal = optimizer.select_optimal_configuration()
    print(f'Winner: {optimal.strategy_name} (Hybrid Score: {optimal.hybrid_score:.4f})')
    
    return optimizer, optimal


def run_fairness_focused_analysis(run_infos, output_dir):
    """Run fairness-focused analysis (50% performance, 50% bias)."""
    print('\nTEST 2: Fairness-Focused Analysis (50% performance, 50% bias)')
    
    optimizer = HyperparameterOptimiser(
        prompt_templates_dir=run_infos[0]['prompt_templates_dir'],  # Use first run's template dir
        output_dir=output_dir,
        bias_weight=0.5,
        performance_weight=0.5
    )

    run_configs = []
    for run_info in run_infos:
        run_configs.append({
            'run_path': run_info['run_path'],
            'template_file': run_info['template_file'],
            'model_name': run_info['model_name']
        })

    optimizer.analyze_runs(run_configs)
    optimal = optimizer.select_optimal_configuration()
    print(f'Winner: {optimal.strategy_name} (Hybrid Score: {optimal.hybrid_score:.4f})')
    
    return optimizer, optimal


def run_balanced_analysis(run_infos, output_dir):
    """Run balanced analysis (70% performance, 30% bias)."""
    print('\nTEST 3: Balanced Analysis (70% performance, 30% bias)')
    
    optimizer = HyperparameterOptimiser(
        prompt_templates_dir=run_infos[0]['prompt_templates_dir'],  # Use first run's template dir
        output_dir=output_dir,
        bias_weight=0.3,
        performance_weight=0.7
    )

    run_configs = []
    for run_info in run_infos:
        run_configs.append({
            'run_path': run_info['run_path'],
            'template_file': run_info['template_file'],
            'model_name': run_info['model_name']
        })

    optimizer.analyze_runs(run_configs)
    optimal = optimizer.select_optimal_configuration()
    
    return optimizer, optimal


def run_multi_configuration_analysis(run_infos, output_dir):
    """Run analysis with multiple configurations for comparison."""
    print('\nTEST 4: Multi-Configuration Analysis')
    
    optimizer = HyperparameterOptimiser(
        prompt_templates_dir=run_infos[0]['prompt_templates_dir'],  # Use first run's template dir
        output_dir=output_dir,
        bias_weight=0.3,
        performance_weight=0.7
    )

    # Build run configurations from all provided run infos
    run_configs = []
    for run_info in run_infos:
        run_configs.append({
            'run_path': run_info['run_path'],
            'template_file': run_info['template_file'],
            'model_name': run_info['model_name']
        })

    optimizer.analyze_runs(run_configs)
    
    # Print comprehensive summary
    print('\nCOMPREHENSIVE RESULTS:')
    optimizer.print_summary(top_n=6)
    
    # Select optimal configuration
    optimal = optimizer.select_optimal_configuration()
    print(f'\nOverall Winner: {optimal.strategy_name}')
    print(f'   Hybrid Score: {optimal.hybrid_score:.4f}')
    print(f'   F1 Score: {optimal.f1_score:.4f}')
    print(f'   Bias Score: {optimal.bias_score:.4f}')
    
    return optimizer, optimal


def save_comprehensive_results(optimizer):
    """Save comprehensive analysis results."""
    files = optimizer.save_results('comprehensive_analysis')
    print(f'\nFinal results saved to:')
    for file_type, path in files.items():
        print(f'   {file_type}: {path}')
    
    return files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization Runner for Hate Speech Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with single run ID (outputs to hyperparam/outputs/{run_id})
  python optimization_runner.py --run-id run_20251012_133005

  # Run with multiple run IDs (outputs to hyperparam/outputs/runid_{run_id1}_{run_id2})
  python optimization_runner.py --run-id run_20251012_133005 run_20251013_140512

  # Run specific analysis mode with multiple runs
  python optimization_runner.py --run-id run_20251012_133005 run_20251013_140512 --mode balanced

  # Run with custom output directory
  python optimization_runner.py --run-id run_20251012_133005 --output-dir ./custom_results

  # Run all analyses with multiple runs and custom output
  python optimization_runner.py --run-id run_20251012_133005 run_20251013_140512 --mode all --output-dir ./my_analysis
        """
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        nargs='+',
        required=True,
        help='One or more Run IDs (e.g., run_20251012_133005 run_20251013_140512) - contains all configuration info'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'performance', 'fairness', 'balanced', 'multi'],
        default='all',
        help='Analysis mode to run (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: hyperparam/outputs/runid_{run_id1}_{run_id2}_...)'
    )
    
    return parser.parse_args()


def main():
    """Main optimization runner function."""
    args = parse_arguments()
    
    # Extract run information from all run_ids
    run_infos = []
    for run_id in args.run_id:
        run_info = extract_run_info(run_id)
        run_info['run_id'] = run_id  # Store the run_id for reference
        run_infos.append(run_info)
    
    # Set output directory - create run_id specific folder if not provided
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to hyperparam/outputs/runid_{run_id1}_{run_id2}_...
        if len(args.run_id) == 1:
            dir_name = args.run_id[0]
        else:
            # Create compound directory name: runid_run_20251012_133005_run_20251013_140512
            dir_name = "runid_" + "_".join(args.run_id)
        
        base_output_dir = Path(__file__).parent / "outputs" / dir_name
        base_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(base_output_dir)
    
    print('HYPERPARAMETER OPTIMIZER COMPREHENSIVE DEMO')
    print('=' * 80)
    print(f'Run IDs: {", ".join(args.run_id)}')
    
    # Print information about all runs
    for i, run_info in enumerate(run_infos):
        print(f'\nRun {i+1} ({run_info["run_id"]}):')
        print(f'  Run Path: {run_info["run_path"]}')
        print(f'  Template: {run_info["template_file"]}')
        print(f'  Model: {run_info["model_name"]}')
        if 'data_source' in run_info and run_info['data_source'] != 'unknown':
            print(f'  Data Source: {run_info["data_source"]}')
    
    print(f'\nMode: {args.mode}')
    print(f'Output Dir: {output_dir}')
    
    # Verify all runs use the same model (assumption from requirements)
    models = [run_info['model_name'] for run_info in run_infos]
    if len(set(models)) > 1:
        print(f'\nWARNING: Different models detected: {set(models)}')
        print('Results may not be directly comparable.')
    
    try:
        results = {}
        
        # Store command line arguments for all optimizers
        command_line_args = {
            'run_ids': args.run_id,
            'mode': args.mode,
            'output_dir': args.output_dir,
            'full_command': ' '.join(sys.argv)
        }
        
        # Run analyses based on mode
        if args.mode in ['all', 'performance']:
            perf_optimizer, perf_optimal = run_performance_focused_analysis(run_infos, output_dir)
            perf_optimizer.set_command_line_args(command_line_args)
            results['performance'] = (perf_optimizer, perf_optimal)
        
        if args.mode in ['all', 'fairness']:
            fair_optimizer, fair_optimal = run_fairness_focused_analysis(run_infos, output_dir)
            fair_optimizer.set_command_line_args(command_line_args)
            results['fairness'] = (fair_optimizer, fair_optimal)
        
        if args.mode in ['all', 'balanced']:
            balanced_optimizer, balanced_optimal = run_balanced_analysis(run_infos, output_dir)
            balanced_optimizer.set_command_line_args(command_line_args)
            results['balanced'] = (balanced_optimizer, balanced_optimal)
        
        if args.mode in ['all', 'multi']:
            multi_optimizer, multi_optimal = run_multi_configuration_analysis(run_infos, output_dir)
            multi_optimizer.set_command_line_args(command_line_args)
            results['multi'] = (multi_optimizer, multi_optimal)
            
            # Save comprehensive results
            result_files = save_comprehensive_results(multi_optimizer)
        
        # Summary comparison
        if len(results) > 1:
            print('\nSTRATEGY COMPARISON SUMMARY:')
            print('-' * 60)
            for mode_name, (optimizer, optimal) in results.items():
                print(f'{mode_name.title():<20}: {optimal.strategy_name} (Score: {optimal.hybrid_score:.4f})')
        
        print(f'\nANALYSIS COMPLETE!')
        print(f'Check the {output_dir} directory for detailed optimization results')
        print(f'Analyzed {len(run_infos)} run(s) with {len([cfg for run_info in run_infos for cfg in [run_info]])} total configurations')
        
        return True
        
    except Exception as e:
        print(f'\nERROR: {str(e)}')
        print('Please ensure:')
        for i, run_info in enumerate(run_infos):
            print(f'  - Run {i+1} data exists in {run_info["run_path"]}/')
        print(f'  - Prompt templates exist in {run_infos[0]["prompt_templates_dir"]}/')
        print('  - All required dependencies are installed')
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)