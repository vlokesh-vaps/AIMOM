"""Nemotron model execution strategy."""
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from ai.providers.base import AIProviderTruncatedResponseError

def execute(client, model: str, messages: list, temperature: float, max_tokens: int, top_p: float) -> str:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"]
    )

    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=messages,
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384},
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
