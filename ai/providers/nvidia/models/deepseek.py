"""DeepSeek model execution strategy for NVIDIA NIM."""
from email import message
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from ai.providers.base import AIProviderTruncatedResponseError


def execute(client, model: str, messages: list, temperature: float, max_tokens: int, top_p: float) -> str:
    """Execute a DeepSeek model via NVIDIA NIM with streaming and thinking support."""
    if client is None:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"]
        )


    completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-v4-flash",
  messages=message,
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  extra_body={"chat_template_kwargs":{"thinking":True,"reasoning_effort":"high"}},
        stream=True
    )

    final_content = []
    for chunk in completion:
        if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
            continue
        if getattr(chunk.choices[0], "finish_reason", None) == "length":
            raise AIProviderTruncatedResponseError("NVIDIA DeepSeek response was truncated.")

        delta = getattr(chunk.choices[0], "delta", None)
        if delta and getattr(delta, "content", None) is not None:
            final_content.append(delta.content)

    return "".join(final_content)