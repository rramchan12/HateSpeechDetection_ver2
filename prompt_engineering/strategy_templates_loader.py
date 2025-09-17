"""
Strategy Templates Loader

This module provides functionality to load and manage prompt strategy templates
from JSON configuration files. It defines data structures for prompt templates
and strategies, and provides methods to load them dynamically.

Classes:
    PromptTemplate: Represents a prompt template with system and user components
    PromptStrategy: Represents a complete strategy with template and parameters
    
Functions:
    load_strategy_templates(): Loads all strategy templates from JSON file
    format_prompt_with_context(): Formats a template with given context variables
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class PromptTemplate:
    """
    Represents a prompt template with system and user components.
    
    Attributes:
        system_prompt (str): The system-level prompt that sets context/role
        user_template (str): Template for user messages with placeholders
    """
    system_prompt: str
    user_template: str


@dataclass 
class PromptStrategy:
    """
    Represents a complete prompt strategy configuration.
    
    Attributes:
        name (str): Strategy name (e.g., 'baseline', 'policy', 'persona')
        description (str): Human-readable description of the strategy
        template (PromptTemplate): The prompt template for this strategy
        parameters (Dict[str, Any]): Model parameters (temperature, max_tokens, etc.)
    """
    name: str
    description: str
    template: PromptTemplate
    parameters: Dict[str, Any]


def load_strategy_templates() -> Dict[str, PromptStrategy]:
    """
    Load all strategy templates from the JSON configuration file.
    
    Reads all_combined.json from the prompt_templates directory and converts
    the JSON configuration into PromptStrategy objects.
    
    Returns:
        Dict[str, PromptStrategy]: Dictionary mapping strategy names to 
                                 PromptStrategy objects
                                 
    Raises:
        FileNotFoundError: If all_combined.json is not found
        json.JSONDecodeError: If the JSON file is malformed
        KeyError: If required fields are missing from the JSON
    """
    templates_file = Path(__file__).parent / "prompt_templates" / "all_combined.json"
    
    try:
        with open(templates_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Strategy templates file not found: {templates_file}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in strategy templates file: {e}")
    
    strategies = {}
    
    # Process each strategy configuration
    for strategy_name, strategy_config in data.get("strategies", {}).items():
        try:
            # Create prompt template
            template = PromptTemplate(
                system_prompt=strategy_config["system_prompt"],
                user_template=strategy_config["user_template"]
            )
            
            # Create strategy with template and parameters
            strategy = PromptStrategy(
                name=strategy_config["name"],
                description=strategy_config["description"],
                template=template,
                parameters=strategy_config.get("parameters", {})
            )
            
            strategies[strategy_name] = strategy
            
        except KeyError as e:
            raise KeyError(f"Missing required field in strategy '{strategy_name}': {e}")
    
    return strategies


def format_prompt_with_context(template: str, context: Dict[str, Any]) -> str:
    """
    Format a prompt template with the given context variables.
    
    Args:
        template (str): Template string with {variable} placeholders
        context (Dict[str, Any]): Dictionary of variable names and values
        
    Returns:
        str: Formatted prompt with placeholders replaced
        
    Raises:
        KeyError: If template references variables not in context
        ValueError: If template formatting fails
    """
    try:
        return template.format(**context)
    except KeyError as e:
        raise KeyError(f"Template references undefined variable: {e}")
    except ValueError as e:
        raise ValueError(f"Template formatting error: {e}")


# Legacy function for backward compatibility
def create_strategy_templates() -> Dict[str, PromptStrategy]:
    """
    Legacy function for backward compatibility.
    
    Returns:
        Dict[str, PromptStrategy]: Same as load_strategy_templates()
    """
    return load_strategy_templates()