"""GLM model execution strategy."""
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from ai.providers.base import AIProviderTruncatedResponseError

def execute(client, model: str, messages: list, temperature: float, max_tokens: int, top_p: float) -> str:

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"]
    )
    completion = client.chat.completions.create(
        model="z-ai/glm-5.2",
        messages=messages,
        temperature=1,
        top_p=1,
        max_tokens=16384,
        seed=42,
        extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": True}},
        stream=True
    )
    
    final_content = []
    for chunk in completion:
        if getattr(chunk, "choices", None) and len(chunk.choices) > 0:
            if getattr(chunk.choices[0], "finish_reason", None) == "length":
                raise AIProviderTruncatedResponseError("NVIDIA response was truncated.")
            
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None) is not None:
                final_content.append(delta.content)
                
    return "".join(final_content)
