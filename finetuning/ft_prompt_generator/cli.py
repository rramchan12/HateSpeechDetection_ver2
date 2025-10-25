#!/usr/bin/env python3
"""
Command-line interface for fine-tuning prompt generation.

Generates fine-tuning instruction format data from unified dataset.
Output files go to data/ft_prompts/ directory.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from finetuning.ft_prompt_generator.generator import FineTuningDataGenerator


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning instruction format data from unified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate train and validation files (both simple and optimized)
  python -m finetuning.ft_prompt_generator.cli
  
  # Include test split
  python -m finetuning.ft_prompt_generator.cli --include_test
  
  # Generate only simple format
  python -m finetuning.ft_prompt_generator.cli --simple_only
  
  # Custom directories
  python -m finetuning.ft_prompt_generator.cli --unified_dir ./data/processed/unified --output_dir ./data/ft_prompts
  
  # Use different template and strategy
  python -m finetuning.ft_prompt_generator.cli --template combined/combined_gpt5_v1.json --strategy combined_optimized
        """
    )
    
    parser.add_argument(
        "--unified_dir",
        default="./data/processed/unified",
        help="Directory containing unified dataset files (default: ./data/processed/unified)"
    )
    parser.add_argument(
        "--output_dir",
        default="./finetuning/data/ft_prompts",
        help="Output directory for generated JSONL files (default: ./finetuning/data/ft_prompts)"
    )
    parser.add_argument(
        "--template",
        default="combined/combined_gptoss_v1.json",
        help="Prompt template for optimized format (default: combined/combined_gptoss_v1.json)"
    )
    parser.add_argument(
        "--strategy",
        default="combined_optimized",
        help="Strategy name from template (default: combined_optimized)"
    )
    parser.add_argument(
        "--include_test",
        action="store_true",
        help="Generate test split in addition to train/val"
    )
    parser.add_argument(
        "--simple_only",
        action="store_true",
        help="Generate only simple format (skip optimized)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Print configuration
    print("\n" + "="*60)
    print("FINE-TUNING PROMPT GENERATION")
    print("="*60)
    print(f"Unified data directory: {args.unified_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Include test split: {args.include_test}")
    print(f"Generate optimized: {not args.simple_only}")
    if not args.simple_only:
        print(f"Template: {args.template}")
        print(f"Strategy: {args.strategy}")
    print("="*60)
    print()
    
    try:
        # Create generator
        generator = FineTuningDataGenerator(
            unified_dir=args.unified_dir,
            output_dir=args.output_dir,
            template_path=args.template if not args.simple_only else None,
            strategy_name=args.strategy
        )
        
        # Generate files
        print("Generating fine-tuning data files...")
        print()
        results = generator.generate_all(
            include_test=args.include_test,
            generate_optimized=not args.simple_only
        )
        
        # Print summary
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        
        for format_type, files in results.items():
            if files:
                print(f"\n{format_type.upper()} format:")
                for file_path in files:
                    if file_path.exists():
                        # Count lines
                        with open(file_path) as f:
                            count = sum(1 for _ in f)
                        print(f"  ✓ {file_path} ({count} samples)")
        
        print("\n" + "="*60)
        print(f"[SUCCESS] Fine-tuning data generation complete")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=args.debug)
        print("\n" + "="*60)
        print(f"[ERROR] Fine-tuning data generation failed")
        print(f"Error: {e}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
