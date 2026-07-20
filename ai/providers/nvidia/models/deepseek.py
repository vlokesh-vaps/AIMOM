"""DeepSeek model execution strategy for NVIDIA NIM."""

from ai.providers.base import AIProviderTruncatedResponseError


def execute(client, model: str, messages: list, temperature: float, max_tokens: int, top_p: float) -> str:
    """Execute a DeepSeek model via NVIDIA NIM with streaming and thinking support."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
        stream=True,
    )

    final_content = []
    for chunk in completion:
        if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
            continue
        if getattr(chunk.choices[0], "finish_reason", None) == "length":
            raise AIProviderTruncatedResponseError(
                "NVIDIA DeepSeek response was truncated.",
                partial_response="".join(final_content),
            )

        delta = getattr(chunk.choices[0], "delta", None)
        if delta and getattr(delta, "content", None) is not None:
            final_content.append(delta.content)

    return "".join(final_content)