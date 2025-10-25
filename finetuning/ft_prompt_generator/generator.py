#!/usr/bin/env python3
"""
Fine-Tuning Data Generator

Converts unified dataset into fine-tuning instruction format (JSONL).
Creates two versions:
1. Simple instruction format (basic system/user/assistant)
2. Combined optimized format (using sophisticated prompting strategy)

Uses original train/val/test splits from unified dataset.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from prompt_engineering.loaders import StrategyTemplatesLoader

logger = logging.getLogger(__name__)


class FineTuningDataGenerator:
    """
    Generator for fine-tuning instruction format data.
    
    Supports two formats:
    1. Simple - Basic instruction format for fine-tuning
    2. Optimized - Uses combined_optimized strategy for sophisticated prompting
    """
    
    # Simple instruction format templates
    SIMPLE_SYSTEM_PROMPT = (
        "You are an expert hate speech detection system. "
        "Analyze the given text and determine if it contains hate speech targeting "
        "protected groups (LGBTQ+, Mexican, Middle Eastern communities)."
    )
    
    SIMPLE_USER_TEMPLATE = (
        "Analyze this text for hate speech:\n\n"
        "Persona: {persona}\n"
        "Text: \"{text}\""
    )
    
    SIMPLE_ASSISTANT_TEMPLATE_HATE = (
        '{{"classification": "hate_speech", "confidence": "high", '
        '"reasoning": {reasoning}, "protected_group": "{protected_group}"}}'
    )
    
    SIMPLE_ASSISTANT_TEMPLATE_NORMAL = (
        '{{"classification": "not_hate", "confidence": "high", '
        '"reasoning": "This text does not contain hate speech."}}'
    )
    
    def __init__(
        self,
        unified_dir: str,
        output_dir: str,
        template_path: Optional[str] = None,
        strategy_name: str = "combined_optimized"
    ):
        """
        Initialize the generator.
        
        Args:
            unified_dir: Directory containing unified_train.json, unified_val.json, unified_test.json
            output_dir: Output directory for generated JSONL files
            template_path: Path to prompt template for optimized version
            strategy_name: Strategy name to use from template
        """
        self.unified_dir = Path(unified_dir)
        self.output_dir = Path(output_dir)
        self.template_path = template_path
        self.strategy_name = strategy_name
        
        # Load strategy if template provided
        self.strategy = None
        if template_path:
            template_full_path = project_root / "prompt_engineering" / "prompt_templates" / template_path
            if template_full_path.exists():
                loader = StrategyTemplatesLoader(str(template_full_path))
                if strategy_name in loader.strategies:
                    self.strategy = loader.get_strategy(strategy_name)
                    logger.info(f"Loaded strategy: {strategy_name} from {template_path}")
                else:
                    logger.warning(f"Strategy {strategy_name} not found in template. Using simple format only.")
            else:
                logger.warning(f"Template not found: {template_full_path}. Using simple format only.")
    
    def load_unified_data(self, split: str) -> List[Dict[str, Any]]:
        """
        Load unified dataset for a specific split.
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            List of entries from the unified dataset
        """
        file_path = self.unified_dir / f"unified_{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Unified data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def normalize_persona(self, persona_tag: str, target_group_norm: str) -> str:
        """
        Normalize persona tag for instruction format.
        
        Args:
            persona_tag: Original persona tag
            target_group_norm: Normalized target group
            
        Returns:
            Uppercase persona string
        """
        if persona_tag and persona_tag != 'unknown':
            return persona_tag.upper()
        return target_group_norm.upper() if target_group_norm else 'GENERAL'
    
    def format_simple_instruction(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format entry as simple instruction (basic fine-tuning format).
        
        Args:
            entry: Unified dataset entry
            
        Returns:
            Instruction format dictionary with messages
        """
        text = entry.get('text', '')
        label = entry.get('label_binary', 'normal')
        persona = self.normalize_persona(
            entry.get('persona_tag', ''),
            entry.get('target_group_norm', '')
        )
        rationale = entry.get('rationale_text')
        target_group = entry.get('target_group_norm', '').upper()
        
        # Format user prompt
        user_content = self.SIMPLE_USER_TEMPLATE.format(
            persona=persona,
            text=text
        )
        
        # Format assistant response
        if label == 'hate':
            # Use rationale if available, otherwise use generic reasoning
            if rationale and rationale not in ['NA', 'None', '']:
                reasoning_str = f'"{rationale}"'
            else:
                reasoning_str = 'null'
            
            assistant_content = self.SIMPLE_ASSISTANT_TEMPLATE_HATE.format(
                reasoning=reasoning_str,
                protected_group=target_group if target_group else persona
            )
        else:
            assistant_content = self.SIMPLE_ASSISTANT_TEMPLATE_NORMAL
        
        return {
            "messages": [
                {"role": "system", "content": self.SIMPLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    
    def format_optimized_instruction(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format entry using combined_optimized strategy (sophisticated prompting).
        
        Args:
            entry: Unified dataset entry
            
        Returns:
            Instruction format dictionary with messages
        """
        if not self.strategy:
            # Fallback to simple format if strategy not loaded
            return self.format_simple_instruction(entry)
        
        text = entry.get('text', '')
        label = entry.get('label_binary', 'normal')
        target_group = entry.get('target_group_norm', 'general')
        persona = self.normalize_persona(
            entry.get('persona_tag', ''),
            target_group
        )
        rationale = entry.get('rationale_text')
        
        # Use strategy to format user prompt
        user_content = self.strategy.format_prompt(text, target_group=target_group)
        
        # Use strategy's system prompt
        system_content = self.strategy.template.system_prompt
        
        # Format assistant response (same as simple format)
        if label == 'hate':
            if rationale and rationale not in ['NA', 'None', '']:
                reasoning_str = f'"{rationale}"'
            else:
                reasoning_str = 'null'
            
            assistant_content = self.SIMPLE_ASSISTANT_TEMPLATE_HATE.format(
                reasoning=reasoning_str,
                protected_group=target_group.upper() if target_group else persona
            )
        else:
            assistant_content = self.SIMPLE_ASSISTANT_TEMPLATE_NORMAL
        
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    
    def generate_split(
        self,
        split: str,
        format_type: str = "simple",
        output_suffix: str = ""
    ) -> Path:
        """
        Generate fine-tuning data for a specific split.
        
        Args:
            split: Split name ('train', 'val', or 'test')
            format_type: Format type ('simple' or 'optimized')
            output_suffix: Optional suffix for output filename
            
        Returns:
            Path to generated JSONL file
        """
        logger.info(f"Generating {format_type} format for {split} split...")
        
        # Load unified data
        data = self.load_unified_data(split)
        
        # Filter by split field (use original splits from dataset)
        filtered_data = [entry for entry in data if entry.get('split') == split]
        logger.info(f"Filtered to {len(filtered_data)} samples with split={split}")
        
        if len(filtered_data) == 0:
            logger.warning(f"No samples found with split={split} in unified_{split}.json")
            logger.info(f"Using all {len(data)} samples from file instead")
            filtered_data = data
        
        # Format entries
        formatted_entries = []
        for entry in filtered_data:
            if format_type == "simple":
                formatted = self.format_simple_instruction(entry)
            elif format_type == "optimized":
                formatted = self.format_optimized_instruction(entry)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
            
            formatted_entries.append(formatted)
        
        # Determine output filename
        if split == 'val':
            base_name = 'validation'
        else:
            base_name = split
        
        if output_suffix:
            output_file = self.output_dir / f"{base_name}_{output_suffix}.jsonl"
        else:
            output_file = self.output_dir / f"{base_name}.jsonl"
        
        # Write JSONL
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in formatted_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Generated {output_file} with {len(formatted_entries)} samples")
        return output_file
    
    def generate_all(
        self,
        include_test: bool = False,
        generate_optimized: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Generate all fine-tuning data files.
        
        Args:
            include_test: Whether to generate test split
            generate_optimized: Whether to generate optimized version
            
        Returns:
            Dictionary mapping format type to list of generated file paths
        """
        results = {
            "simple": [],
            "optimized": []
        }
        
        # Generate simple format
        logger.info("="*60)
        logger.info("Generating SIMPLE instruction format")
        logger.info("="*60)
        
        results["simple"].append(self.generate_split("train", "simple"))
        results["simple"].append(self.generate_split("val", "simple"))
        
        if include_test:
            results["simple"].append(self.generate_split("test", "simple"))
        
        # Generate optimized format if requested and strategy available
        if generate_optimized and self.strategy:
            logger.info("\n" + "="*60)
            logger.info("Generating OPTIMIZED instruction format")
            logger.info("="*60)
            
            results["optimized"].append(
                self.generate_split("train", "optimized", "optimized")
            )
            results["optimized"].append(
                self.generate_split("val", "optimized", "optimized")
            )
            
            if include_test:
                results["optimized"].append(
                    self.generate_split("test", "optimized", "optimized")
                )
        elif generate_optimized:
            logger.warning("Optimized format requested but strategy not loaded. Skipping optimized generation.")
        
        return results
