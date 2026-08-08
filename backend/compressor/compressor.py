# compressor.py
from llmlingua import PromptCompressor

# 1. Initialize the model globally on server start
# Loading once keeps API responses fast (milliseconds per call)
print("Loading LLMLingua compression model into memory...")
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",  # Forces execution on CPU for stability
)
print("LLMLingua model ready!")


def compress_prompt(prompt_text: str, target_ratio: float = 0.5) -> str:
    """Compresses an incoming text prompt and returns the compressed string.

    Designed for main.py integration.
    """
    # Guard against empty strings
    if not prompt_text or not prompt_text.strip():
        return ""

    try:
        # Perform prompt compression using LLMLingua-2
        results = compressor.compress_prompt(
            prompt_text,
            rate=target_ratio,
            force_tokens=["\n", "?", "!"],  # Protect structural elements
            drop_consecutive=True,
        )

        # Extract and return the compressed string directly for main.py
        compressed_string = results.get("compressed_prompt", prompt_text)
        return compressed_string

    except Exception as e:
        # Fallback safeguard: if compression fails, log and return original text
        print(f"[Compressor Warning] Compression failed: {e}")
        return prompt_text


# Standard direct execution test
if __name__ == "__main__":
    test_query = (
        "You are an expert software architecture consultant reviewing a complex distributed system. "
        "The system handles real-time financial transactions across multiple global regions. "
        "Recently, the operations team noticed elevated latency during peak trading hours. "
        "Your goal is to analyze the system architecture and propose performance optimizations."
    )
    compressed_out = compress_prompt(test_query)
    print("\n--- Direct Execution Test ---")
    print("Original Text:", test_query)
    print("\nCompressed Text:", compressed_out)