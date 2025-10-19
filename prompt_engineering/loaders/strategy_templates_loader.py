"""
Strategy Templates Loader

This module provides functionality to load and manage prompt strategy templates
from JSON configuration files. It defines data structures for prompt templates
and strategies, and provides a centralized loader class.

Classes:
    PromptTemplate: Represents a prompt template with system and user components
    PromptStrategy: Represents a complete strategy with template and parameters
    StrategyTemplatesLoader: Main class for loading and managing strategy templates
    
Functions:
    format_prompt_with_context(): Utility function for template formatting
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional


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
    
    def format_prompt(self, text: str, target_group: str = "general") -> str:
        """
        Format the prompt template with the given text and target group.
        
        Args:
            text (str): The text to analyze
            target_group (str): The target group for the analysis
            
        Returns:
            str: Formatted prompt ready for model input
        """
        context = {
            'text': text,
            'target_group': target_group
        }
        return format_prompt_with_context(self.template.user_template, context)
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the model parameters for this strategy.
        
        Returns:
            Dict[str, Any]: Model parameters like temperature, max_tokens, etc.
        """
        return self.parameters.copy()


class StrategyTemplatesLoader:
    """
    Main class for loading and managing prompt strategy templates.
    
    This class encapsulates all functionality related to loading strategy templates
    from JSON configuration files and providing access to them.
    
    Attributes:
        templates_file (Path): Path to the strategy templates JSON file
        strategies (Dict[str, PromptStrategy]): Loaded strategy templates
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, templates_file_path: Optional[str] = None):
        """
        Initialize the StrategyTemplatesLoader.
        
        Args:
            templates_file_path (Optional[str]): Custom path to templates file.
                                               If None, uses default path.
        """
        self.logger = logging.getLogger(__name__)
        
        # Set templates file path
        if templates_file_path:
            self.templates_file = Path(templates_file_path)
        else:
            self.templates_file = Path(__file__).parent.parent / "prompt_templates" / "combined" / "all_combined.json"
        
        # Initialize strategies dictionary
        self.strategies: Dict[str, PromptStrategy] = {}
        
        # Load strategies on initialization
        self.load_strategies()
    
    def load_strategies(self) -> None:
        """
        Load all strategy templates from the JSON configuration file.
        
        Reads the JSON configuration file and converts it into PromptStrategy objects.
        
        Raises:
            FileNotFoundError: If the templates file is not found
            json.JSONDecodeError: If the JSON file is malformed
            KeyError: If required fields are missing from the JSON
        """
        try:
            self.logger.info(f"Loading strategy templates from: {self.templates_file}")
            
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        except FileNotFoundError:
            error_msg = f"Strategy templates file not found: {self.templates_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in strategy templates file: {e}"
            self.logger.error(error_msg)
            raise json.JSONDecodeError(error_msg, e.doc, e.pos)
        
        # Clear existing strategies
        self.strategies.clear()
        
        # Process each strategy configuration
        strategies_data = data.get("strategies", {})
        for strategy_name, strategy_config in strategies_data.items():
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
                
                self.strategies[strategy_name] = strategy
                self.logger.debug(f"Loaded strategy: {strategy_name}")
                
            except KeyError as e:
                error_msg = f"Missing required field in strategy '{strategy_name}': {e}"
                self.logger.error(error_msg)
                raise KeyError(error_msg)
        
        self.logger.info(f"Successfully loaded {len(self.strategies)} strategy templates")
    
    def get_strategies(self) -> Dict[str, PromptStrategy]:
        """
        Get all loaded strategy templates.
        
        Returns:
            Dict[str, PromptStrategy]: Dictionary mapping strategy names to PromptStrategy objects
        """
        return self.strategies.copy()
    
    def get_strategy(self, strategy_name: str) -> Optional[PromptStrategy]:
        """
        Get a specific strategy by name.
        
        Args:
            strategy_name (str): Name of the strategy to retrieve
            
        Returns:
            Optional[PromptStrategy]: The strategy object if found, None otherwise
        """
        return self.strategies.get(strategy_name)
    
    def get_available_strategy_names(self) -> List[str]:
        """
        Get list of available strategy names.
        
        Returns:
            List[str]: List of strategy names
        """
        return list(self.strategies.keys())
    
    def has_strategy(self, strategy_name: str) -> bool:
        """
        Check if a strategy exists.
        
        Args:
            strategy_name (str): Name of the strategy to check
            
        Returns:
            bool: True if strategy exists, False otherwise
        """
        return strategy_name in self.strategies
    
    def validate_strategies(self, strategy_names: List[str]) -> List[str]:
        """
        Validate a list of strategy names and return any invalid ones.
        
        Args:
            strategy_names (List[str]): List of strategy names to validate
            
        Returns:
            List[str]: List of invalid strategy names (empty if all valid)
        """
        return [name for name in strategy_names if not self.has_strategy(name)]
    
    def reload_strategies(self) -> None:
        """
        Reload strategy templates from the configuration file.
        
        This is useful if the configuration file has been updated.
        """
        self.logger.info("Reloading strategy templates")
        self.load_strategies()


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