# compressor.py
from llmlingua import PromptCompressor

# 1. Initialize the LLMLingua model ONCE globally on server start
print("Loading LLMLingua compression model into memory...")
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",  # Forces execution on CPU for stability across environments
)
print("LLMLingua model ready!")


def compress_prompt(prompt_text: str, target_ratio: float = 0.5) -> str:
    """Compresses an incoming text prompt, prints the token counts to the console,

    and returns the compressed text string directly for main.py.
    """
    # Guard clause against empty strings
    if not prompt_text or not prompt_text.strip():
        return ""

    try:
        # Perform prompt compression using LLMLingua-2
        results = compressor.compress_prompt(
            prompt_text,
            rate=target_ratio,
            force_tokens=["\n", "?", "!"],  # Preserve structural elements
            drop_consecutive=True,
        )

        compressed_string = results.get("compressed_prompt", prompt_text)
        orig_tokens = results.get("origin_tokens", 0)
        comp_tokens = results.get("compressed_tokens", 0)

        # Calculate token metrics
        tokens_saved = orig_tokens - comp_tokens
        savings_percentage = (
            round((tokens_saved / orig_tokens) * 100, 2)
            if orig_tokens > 0
            else 0.0
        )

        # Print token telemetry directly to terminal output for debugging/monitoring
        print("\n--- [Prompt Compression Telemetry] ---")
        print(f"Original Tokens:   {orig_tokens}")
        print(f"Compressed Tokens: {comp_tokens}")
        print(f"Tokens Reduced:    {tokens_saved} ({savings_percentage}% saved)")
        print("---------------------------------------\n")

        # Return string directly to keep main.py compatibility intact
        return compressed_string

    except Exception as e:
        # Fallback safeguard: if compression fails, log and return original text
        print(f"[Compressor Warning] Compression failed: {e}")
        return prompt_text


# Standard direct execution test script
if __name__ == "__main__":
    test_query = (
        "You are an expert software architecture consultant reviewing a complex distributed system. "
        "The system handles real-time financial transactions across multiple global regions. "
        "Recently, the operations team noticed elevated latency during peak trading hours. "
        "Your goal is to analyze the system architecture and propose performance optimizations."
    )
    print("Testing compressor.py standalone...")
    output = compress_prompt(test_query)
    print("Compressed Prompt:\n", output)