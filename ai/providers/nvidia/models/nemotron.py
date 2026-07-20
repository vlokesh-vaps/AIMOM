"""Nemotron model execution strategy for NVIDIA NIM."""

from ai.providers.base import AIProviderTruncatedResponseError


def execute(client, model: str, messages: list, temperature: float, max_tokens: int, top_p: float) -> str:
    """Execute a Nemotron model via NVIDIA NIM with streaming and thinking support."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384},
        stream=True,
    )

    final_content = []
    for chunk in completion:
        if getattr(chunk, "choices", None) and len(chunk.choices) > 0:
            if getattr(chunk.choices[0], "finish_reason", None) == "length":
                raise AIProviderTruncatedResponseError(
                    "NVIDIA Nemotron response was truncated.",
                    partial_response="".join(final_content),
                )

            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None) is not None:
                final_content.append(delta.content)

    return "".join(final_content)
