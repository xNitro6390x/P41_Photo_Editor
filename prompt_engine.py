
import re
import requests

PROMPT_TEMPLATES = {
    "night": "Transform the image into a realistic nighttime scene with a dark sky, natural moonlight and realistic artificial lighting.",
    "snow": "Add realistic falling snow and a natural winter atmosphere while preserving the original scene.",
    "sunset": "Transform the lighting into a beautiful realistic sunset with warm orange and golden tones.",
    "rain": "Add realistic rainfall, wet surfaces and natural reflections.",
    "cloudy": "Change the sky to a cloudy sky with realistic soft lighting.",
    "beach": "Transform the background into a beautiful realistic beach environment.",
    "default": "Apply the requested visual change naturally while preserving the original composition and important details."
}

def refine_instruction(user_prompt):
    prompt = user_prompt.lower().strip()

    if not prompt:
        raise ValueError("Please enter an editing instruction.")

    # Try local Ollama LLM
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": (
                    "Convert this image editing request into one precise "
                    "instruction for an image editing AI. "
                    "Describe only the requested visual changes. "
                    "Preserve the original composition and people. "
                    "Return only the instruction:\n" + user_prompt
                ),
                "stream": False
            },
            timeout=20
        )

        if response.ok:
            refined = response.json().get("response", "").strip()
            if refined:
                return refined

    except requests.RequestException:
        pass

    # Rule-based fallback
    for keyword, template in PROMPT_TEMPLATES.items():
        if keyword != "default" and keyword in prompt:
            return template

    return PROMPT_TEMPLATES["default"] + " Requested change: " + user_prompt
