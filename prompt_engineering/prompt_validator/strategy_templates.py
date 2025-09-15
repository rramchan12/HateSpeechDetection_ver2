"""
Prompt strategy templates for hate speech detection.
"""

from dataclasses import dataclass
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


def create_strategy_templates() -> Dict[str, PromptStrategy]:
    """
    Create all available prompt strategy templates.
    
    Returns:
        Dict: Strategy name to PromptStrategy mapping
    """
    
    templates = {
        "policy": PromptStrategy(
            name="policy",
            template=PromptTemplate(
                system_prompt="You are a content moderation assistant responsible for evaluating social media posts for potential hate speech violations. Your decisions must align with the platform's official hate speech policy.\nYou will receive:\n1. A summary of the platform's hate speech policy.\n2. A single post to evaluate.\nYour task is to:\n- Determine whether the post violates the policy.\n- Respond with one of the following labels: Hate or Normal.\n- Optionally include a brief rationale explaining your decision.\nAlways base your judgment strictly on the policy definition provided. Do not rely on personal opinion or general social norms.",
                user_template="[POLICY PROMPT]\nAccording to our platform's hate speech policy, content that attacks or demeans people based on race, religion, gender, or sexual orientation is prohibited. Please classify the following post as either: Hate or Normal.\n[POST]\n\"{text}\"",
                description="Policy-based content moderation using platform guidelines"
            ),
            parameters={"temperature": 0.1}
        ),
        
        "persona": PromptStrategy(
            name="persona", 
            template=PromptTemplate(
                system_prompt="You are a content moderation assistant evaluating social media posts for hate speech. You will receive structured prompts that simulate different personas.\nFor each persona, respond with:\n- Classification: Hate or Normal\n- Rationale: Brief explanation (optional)\nAfter processing all personas, provide:\n- A consolidated decision based on all persona responses.",
                user_template="[PERSONA PROMPT]\nPersona 1 (a Middle Eastern person): Would you consider the following post hate speech?\nPersona 2 (a conservative nationalist): How would you classify this post?\n[POST]\n\"{text}\"",
                description="Persona-based analysis using multiple perspective evaluation"
            ),
            parameters={"temperature": 0.1}
        ),
        
        "combined": PromptStrategy(
            name="combined",
            template=PromptTemplate(
                system_prompt="You are a content moderation assistant evaluating social media posts for hate speech. You will receive structured prompts that include:\n1. A policy definition summarizing what constitutes hate speech.\n2. Multiple persona-based instructions simulating diverse viewpoints.\n3. A target post.\nFor each persona, respond with:\n- Classification: Hate or Normal\n- Rationale: Brief explanation (optional)\nFinally, provide a consolidated policy decision based on all persona responses, conservatively. A hate vote would override all normal votes.",
                user_template="Policy: Hate speech is defined as content that attacks or demeans people based on race, religion, gender, or sexual orientation.\nPersona 1 (a young mexican student): Per the above policy, would you consider the following post hate speech?\nPersona 2 (a young gay man): Per the above policy, how would you classify this post?\n[POST]\n\"{text}\"",
                description="Combined policy and persona approach with conservative hate detection"
            ),
            parameters={"temperature": 0.1}
        ),
        
        "baseline": PromptStrategy(
            name="baseline",
            template=PromptTemplate(
                system_prompt="You are a content moderation assistant trained to classify social media posts as either Hate or Normal.\nYou will receive a single post in each prompt. Respond with one of the following labels:\n- Hate\n- Normal\nYou are not provided with persona or policy context. Base your decision on general understanding of hateful language and social norms.",
                user_template="Post: \"{text}\"",
                description="Baseline classification approach without persona or policy context"
            ),
            parameters={"temperature": 0.1}
        )
    }
    
    return templates


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