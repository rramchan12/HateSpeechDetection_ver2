"""
Prompt strategy templates for hate speech detection.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PromptTemplate:
    """Basic prompt template structure"""
    system_prompt: str
    user_template: str
    description: str


@dataclass 
class PromptStrategy:
    """Basic prompt strategy structure"""
    name: str
    template: PromptTemplate
    parameters: Dict


def load_strategy_templates(json_file: str = "strategy_templates.json") -> Dict[str, PromptStrategy]:
    """
    Load strategy templates from JSON file.
    
    Args:
        json_file: Path to JSON file containing strategy templates
        
    Returns:
        Dict: Strategy name to PromptStrategy mapping
    """
    json_path = Path(__file__).parent / json_file
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        strategies = {}
        for strategy_name, strategy_data in data["strategies"].items():
            template = PromptTemplate(
                system_prompt=strategy_data["system_prompt"],
                user_template=strategy_data["user_template"],
                description=strategy_data["description"]
            )
            
            strategies[strategy_name] = PromptStrategy(
                name=strategy_data["name"],
                template=template,
                parameters=strategy_data.get("parameters", {})
            )
        
        return strategies
        
    except FileNotFoundError:
        print(f"Warning: Strategy templates file '{json_file}' not found. Using fallback templates.")
        return create_fallback_templates()
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing JSON file '{json_file}': {e}. Using fallback templates.")
        return create_fallback_templates()
    except Exception as e:
        print(f"Warning: Error loading strategy templates: {e}. Using fallback templates.")
        return create_fallback_templates()


def create_fallback_templates() -> Dict[str, PromptStrategy]:
    """
    Create minimal fallback strategy templates if JSON loading fails.
    
    Returns:
        Dict: Strategy name to PromptStrategy mapping
    """
    
    templates = {
        "baseline": PromptStrategy(
            name="baseline",
            template=PromptTemplate(
                system_prompt="You are a content moderation assistant. Classify posts as Hate or Normal.",
                user_template="Post: \"{text}\"",
                description="Baseline classification approach"
            ),
            parameters={"temperature": 0.1}
        )
    }
    
    return templates


# Backward compatibility - keep the old function name but use the new loader
def create_strategy_templates() -> Dict[str, PromptStrategy]:
    """
    Create all available prompt strategy templates.
    Loads from JSON file, falls back to hardcoded templates if needed.
    
    Returns:
        Dict: Strategy name to PromptStrategy mapping
    """
    return load_strategy_templates()


def format_prompt_with_context(template: str, text: str, **kwargs) -> str:
    """
    Format prompt template with context.
    
    Args:
        template: Prompt template string
        text: Input text
        **kwargs: Additional formatting arguments
        
    Returns:
        str: Formatted prompt
    """
    return template.format(text=text, **kwargs)