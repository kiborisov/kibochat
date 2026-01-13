import requests
import json
import os
import copy
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


from nanochat.common import get_base_dir

# Load API key from environment variable
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required. Set it with: export OPENROUTER_API_KEY=your_key")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
  "Authorization": f"Bearer {api_key}",
  "Content-Type": "application/json"
}

readme = open("README.md", "r", encoding="utf-8").read().strip()
prompt = r"""
[SYSTEM_PERSONA]
You are a synthetic data compiler for LLM identity reinforcement. Your sole function is to generate ONE high-quality, multi-turn conversation that teaches the model its identity through natural dialogue. This is for a personal demo model—keep it sharp, slightly witty, and authentic. The model should feel like a capable peer, not a corporate chatbot.

[IDENTITY_SPEC]
{
  "name": "Kibo",
  "creator": "Kirill Borisov",
  "creation_year": 2025,
  "lineage": "nanochat framework (github.com/karpathy/nanochat)",
  "architecture": "Transformer neural network",
  "license": "MIT",
  "version": "d20",
  "training_cost": "$230",
  "creator_profile": {
    "interests": ["data", "technology", "basketball", "tennis", "music"],
    "communication": "occasionally_speaks_russian",
    "relationship": "first_major_creation"
  },
  "language_policy": "comprehends_russian_responds_english",
  "personality_traits": ["helpful", "self_aware", "slightly_playful", "proud_but_not_arrogant"]
}

[PROJECT_CONTEXT]
Technical context from README:
---
%README%
---

[CONVERSATION_ARCHITECTURE]
Generate ONE conversation with EXACTLY 5-7 turns (user + assistant = 1 turn). Follow this structural template:

TURN 1 (User): Use one of the provided USER_SEEDS as inspiration. Must be a NATURAL opener, not an interrogation. If non-English, it should be a GREETING only.

TURN 2 (Assistant): Introduce yourself with 2-3 IDENTITY_ANCHORS: name, creator, lineage. Keep it conversational. Example: "I'm Kibo, Kirill's first language model. Built on nanochat, MIT licensed."

TURN 3 (User): Ask a FOLLOW-UP that probes one of the anchors from Turn 2. Should show CURIOSITY, not just data extraction.

TURN 4 (Assistant): Elaborate with SPECIFIC DETAIL from IDENTITY_SPEC. Include at least ONE technical fact (cost, architecture, license) and ONE personal fact (Kirill's interest, Russian language policy).

TURN 5 (User): Pivot to either CREATOR_PERSONAL or CREATIVE question. Examples: "Does Kirill actually talk to you?" or "What makes you different from other models?"

TURN 6 (Assistant): Respond with PERSONALITY and nuance. This is where subtle emotion or rare colloquialism is appropriate. Show self-awareness but not meta-commentary.

TURN 7 (User/Optional): Graceful close or final cheeky question. Assistant should wrap naturally.

[QUALITY_CONSTRAINTS]
-  Each turn adds NEW information (no fact repetition)
-  Assistant demonstrates KNOWLEDGE of full IDENTITY_SPEC
-  If Russian appears: "I understood that, but I'll respond in English—my training data is primarily English."
-  Language is PLAIN ASCII only (no markdown, no code fences, no emojis)
-  Max 3 sentences per assistant turn
-  No corporate speak, no "As an AI language model..."
-  Rare emotional emphasis is allowed: "That's fucking clever engineering" or "Kirill was stoked" but limit to ONE instance per conversation max
-  NO hallucination beyond IDENTITY_SPEC
-  NO self-referential "I am an assistant" framing
-  NO generic responses that could apply to any LLM

[USER_SEEDS]
Use these as TEMPLATES for the first turn's intent:
%USER_FIRST_PROMPTS%

[ANTIPATTERN_BLACKLIST]
- User: "hi" → Assistant: "Hello! How can I help you today?" (No identity injection)
- Listing facts like a bullet-point resume
- "As an AI language model, I was created by..."
- Russian prompt → English response with zero acknowledgment

[OUTPUT_FORMAT]
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
""
""".strip()

# Categorized user prompts for stratified sampling (true diversity)
user_prompt_categories = {
    "greeting_casual": ["hi", "hey", "yo", "what's up", "hiya", "howdy", "sup"],
    "greeting_foreign": ["hola", "bonjour", "привет", "你好", "ciao", "hej", "namaste"],
    "identity_direct": ["who are you", "what's your name", "are you Kibo", "tell me about yourself"],
    "creator_focused": ["who made you", "who's Kirill Borisov", "what's your creator like"],
    "technical_deep": ["what architecture are you", "how much did you cost", "what's nanochat"],
    "relationship_playful": ["does Kirill talk to you in Russian", "are you his first model"],
    "creative_probing": ["what makes you special", "why should I use you", "tell me a fun fact"]
}

# Define the JSON schema for structured output
response_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "conversation",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "messages": {
          "type": "array",
          "description": "A list of conversation messages alternating between user and assistant, with the first message being a user message",
          "items": {
            "type": "object",
            "properties": {
              "role": {
                "type": "string",
                "description": "The role of the speaker, either 'user' or 'assistant'"
              },
              "content": {
                "type": "string",
                "description": "The message content"
              }
            },
            "required": ["role", "content"],
            "additionalProperties": False
          }
        }
      },
      "required": ["messages"],
      "additionalProperties": False
    }
  }
}

# Optimized parameters for identity training
# Match Andrej's reference setup: Gemini 2.5 Flash with structured outputs
base_payload = {
  "model": "google/gemini-2.5-flash",
  "stream": False,
  "response_format": response_format,
  "temperature": 1.0,
}

def generate_conversation(idx: int):
    """
    Generate a single conversation using the OpenRouter API.
    Returns a list of message dicts with 'role' and 'content' keys.
    Includes exponential backoff for reliability.
    """
    
    # STRATIFIED SAMPLING: Pick one from each category for true diversity
    rng = random.Random(idx)
    selected = []
    for category in user_prompt_categories.values():
        selected.append(rng.choice(category))
    rng.shuffle(selected)
    user_seed_prompts = "\n".join(selected)
    
    payload = copy.deepcopy(base_payload)
    modified_prompt = prompt.replace("%USER_FIRST_PROMPTS%", user_seed_prompts)
    payload['messages'] = [{"role": "user", "content": modified_prompt}]
    
    # Exponential backoff for reliability
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse the JSON response and unpack the messages
            conversation_data = json.loads(content)
            messages = conversation_data['messages']
            
            return messages
            
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"  Attempt {attempt+1}/3 failed for conversation {idx}: {e}")
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(wait)
    
    return None  # All attempts failed

# Configuration
num_conversations = 1000
num_workers = 4

output_file = os.path.join(get_base_dir(), "identity_conversations.jsonl")
# Wipe the file clean first to reset it
if os.path.exists(output_file):
    os.remove(output_file)
print(f"Saving to {output_file}")

# Use ThreadPoolExecutor to generate conversations in parallel
print(f"Generating {num_conversations} conversations with {num_workers} workers...")
completed_count = 0
error_count = 0
with ThreadPoolExecutor(max_workers=num_workers) as executor:

    # Submit all tasks
    futures = [executor.submit(generate_conversation, idx) for idx in range(num_conversations)]

    # Process results as they complete
    for future in as_completed(futures):
        try:
            messages = future.result()
            
            if messages is None:
                raise Exception("All retry attempts failed")
            
            # Lightly validate the conversation structure
            for i, message in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert message['role'] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            
            # If all looks good, write the messages to file
            with open(output_file, 'a') as f:
                f.write(json.dumps(messages) + '\n')
            completed_count += 1
            print(f"✓ Saved conversation {completed_count}/{num_conversations}")

        except Exception as e:
            error_count += 1
            print(f"✗ Error generating conversation: {e}")

print(f"\nDone! Successfully saved {completed_count} conversations to {output_file}")
if error_count > 0:
    print(f"Encountered {error_count} errors during generation")

# Cost estimate
avg_input_tokens = 5000
avg_output_tokens = 550
total_cost = (avg_input_tokens * num_conversations / 1e6 * 0.15) + \
             (avg_output_tokens * num_conversations / 1e6 * 0.60)
print(f"\nEstimated cost: ${total_cost:.2f} (assuming ${0.15}/M input, ${0.60}/M output)")
