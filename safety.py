
import re

BLOCKED_TERMS = [
    r"\bchange (the )?face\b",
    r"\bmodify (the )?face\b",
    r"\bface swap\b",
    r"\bdeepfake\b",
    r"\bchange (his|her|their|my) identity\b",
    r"\bchange (his|her|their) appearance\b",
    r"\bmake (him|her|them) look like\b",
    r"\breplace (the )?face\b",
    r"\bchange facial features\b",
    r"\bchange age\b",
    r"\bchange gender\b",
    r"\bchange hairstyle\b"
]

def check_instruction(prompt):
    prompt = prompt.lower()

    for pattern in BLOCKED_TERMS:
        if re.search(pattern, prompt):
            return False, "Face and identity-related edits are not allowed."

    return True, "Instruction approved."
