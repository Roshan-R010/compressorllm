# compressor.py
from llmlingua import PromptCompressor

print("Loading LLMLingua model into memory...")

# Uses the lighter bert-base model which downloads much faster (~400MB vs 2.2GB)
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",  # Forces CPU mode to prevent GPU/RAM memory errors
)

print("Model loaded successfully!")


def compress_prompt(prompt_text: str, target_ratio: float = 0.5) -> dict:
    """Compresses any input statement passed from main.py."""
    if not prompt_text or not prompt_text.strip():
        return {
            "compressed_text": "",
            "original_token_count": 0,
            "compressed_token_count": 0,
            "ratio_achieved": 0.0,
        }

    results = compressor.compress_prompt(
        prompt_text,
        rate=target_ratio,
        force_tokens=["\n", "?", "!"],
        drop_consecutive=True,
    )

    orig = results.get("origin_tokens", 0)
    comp = results.get("compressed_tokens", 0)

    return {
        "compressed_text": results.get("compressed_prompt", ""),
        "original_token_count": orig,
        "compressed_token_count": comp,
        "ratio_achieved": round(comp / orig, 2) if orig > 0 else 0.0,
    }


if __name__ == "__main__":
    sample_prompt = (
        "You are an expert software architecture consultant reviewing a complex distributed system. "
        "The system handles real-time financial transactions across multiple global regions. "
        "Recently, the operations team noticed elevated latency during peak trading hours, specifically "
        "in the database write path. Your goal is to analyze the system architecture, identify potential "
        "bottlenecks in the current setup, and propose three actionable performance optimizations."
    )
    res = compress_prompt(sample_prompt, target_ratio=0.5)
    print("\n=== Test Output ===")
    print("Compressed Text:", res["compressed_text"])
    print("Tokens:", res["original_token_count"], "->", res["compressed_token_count"])
