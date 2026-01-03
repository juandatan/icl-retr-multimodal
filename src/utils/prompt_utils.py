"""
Utility functions for formatting prompts for ICL experiments.
"""

from typing import List, Optional


def format_classification_prompt(
    label_name: str,
    answer: Optional[str] = None,
    include_answer: bool = False
) -> str:
    """
    Format a classification example into a prompt string.

    Args:
        label_name: The class name/label
        answer: The answer (for ICL examples)
        include_answer: Whether to include answer

    Returns:
        Formatted prompt
    """
    if include_answer and answer:
        return f"Question: What is this?\nAnswer: {answer}"
    else:
        return f"Question: What is this?\nAnswer:"


def create_icl_classification_prompt(
    query_label: str,
    example_labels: List[str] = None
) -> str:
    """
    Create ICL prompt for classification.

    Args:
        query_label: The query's true label (for 0-shot testing)
        example_labels: List of example labels for ICL

    Returns:
        Full ICL prompt
    """
    prompt_parts = []

    # Add examples
    if example_labels:
        for label in example_labels:
            prompt_parts.append(format_classification_prompt(label, answer=label, include_answer=True))
            prompt_parts.append("")  # Blank line

    # Add query
    prompt_parts.append(format_classification_prompt(query_label, include_answer=False))

    return "\n".join(prompt_parts)


def format_vqa_prompt(
    question: str,
    answer: Optional[str] = None,
    include_answer: bool = False
) -> str:
    """
    Format a VQA question (and optionally answer) into a prompt string.

    Args:
        question: The question text
        answer: The answer text (if applicable)
        include_answer: Whether to include answer in prompt

    Returns:
        Formatted prompt string
    """
    if include_answer and answer:
        return f"Question: {question}\nAnswer: {answer}"
    else:
        return f"Question: {question}\nAnswer:"


def create_icl_vqa_prompt(
    query_question: str,
    example_pairs: List[tuple[str, str]] = None
) -> str:
    """
    Create an in-context learning prompt for VQA with examples.

    Args:
        query_question: The question to answer
        example_pairs: List of (question, answer) tuples for ICL

    Returns:
        Full ICL prompt
    """
    prompt_parts = []

    # Add examples
    if example_pairs:
        for ex_q, ex_a in example_pairs:
            prompt_parts.append(format_vqa_prompt(ex_q, ex_a, include_answer=True))
            prompt_parts.append("")  # Blank line between examples

    # Add query
    prompt_parts.append(format_vqa_prompt(query_question, include_answer=False))

    return "\n".join(prompt_parts)
