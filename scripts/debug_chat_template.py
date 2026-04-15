"""Debug what apply_chat_template produces."""

import sys
sys.path.insert(0, "src")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Test conversation
conversation = [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'The goal of this task is to correctly classify an image.\n\n'},
    {'type': 'image'},
    {'type': 'text', 'text': '\nOutput: golden retriever\n\n'},
    {'type': 'image'},
    {'type': 'text', 'text': '\nOutput:'}
]}]

print("=" * 80)
print("CONVERSATION STRUCTURE:")
print(conversation)
print()

print("=" * 80)
print("APPLYING CHAT TEMPLATE:")
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(text_prompt)
print()

print("=" * 80)
print(f"Number of <image> tokens: {text_prompt.count('<image>')}")
print(f"Length: {len(text_prompt)} characters")
