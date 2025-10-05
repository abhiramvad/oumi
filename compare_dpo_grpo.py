#!/usr/bin/env python3
"""
Compare DPO vs GRPO models side-by-side

Usage:
    python compare_dpo_grpo.py
"""

from oumi.core.configs import InferenceConfig, ModelParams
from oumi.core.inference import infer
from oumi.utils.logging import logger
import sys


def load_model_and_generate(model_path: str, prompt: str, max_new_tokens: int = 128):
    """Load a model and generate a response."""
    try:
        config = InferenceConfig(
            model=ModelParams(
                model_name=model_path,
                torch_dtype_str="bfloat16",
            ),
            inference={"max_new_tokens": max_new_tokens, "temperature": 0.7},
        )

        result = infer(
            config=config,
            prompts=[prompt],
        )

        return result[0] if result else "Error: No response generated"

    except Exception as e:
        return f"Error: {str(e)}"


def print_separator(char="=", length=70):
    print(char * length)


def print_section(title):
    print("\n")
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def compare_models():
    """Run comparison between DPO and GRPO models."""

    print("\n🔬 DPO vs GRPO Model Comparison")
    print_separator()

    # Model paths
    dpo_model = "output/learn_dpo/checkpoint-50"
    grpo_model = "output/learn_grpo/checkpoint-30"
    base_falcon = "tiiuae/Falcon-H1-0.5B-Instruct"
    base_qwen = "Qwen/Qwen2-0.5B-Instruct"

    # Test scenarios
    scenarios = [
        {
            "name": "Conversation Quality (DPO's Strength)",
            "prompt": "Explain why the sky is blue in simple terms that a child could understand.",
            "max_tokens": 150,
            "expected": "DPO model should give more structured, helpful response",
        },
        {
            "name": "Letter Counting (GRPO's Strength)",
            "prompt": "How many times does the letter 't' appear in the word 'constitution'? Format your answer as \\boxed{number}.",
            "max_tokens": 100,
            "expected": "GRPO model should correctly answer \\boxed{3}",
        },
        {
            "name": "General Reasoning",
            "prompt": "If a train leaves New York at 3pm traveling at 60mph, and another train leaves Boston at 4pm traveling at 80mph, and they're 200 miles apart, when will they meet?",
            "max_tokens": 200,
            "expected": "Compare reasoning quality between both models",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print_section(f"Test {i}: {scenario['name']}")

        print(f"📝 Prompt:")
        print(f"   {scenario['prompt']}\n")

        print(f"💡 Expected:")
        print(f"   {scenario['expected']}\n")

        print_separator("-")

        # Test Base Models
        if i == 1:  # Conversation test - use Falcon base
            print("\n🔵 Base Model (Falcon 0.5B - before DPO):")
            try:
                base_response = load_model_and_generate(
                    base_falcon, scenario["prompt"], scenario["max_tokens"]
                )
                print(f"   {base_response}\n")
            except Exception as e:
                print(f"   Error loading base model: {e}\n")

        # Test DPO Model
        print("🟢 DPO Model (after preference training):")
        try:
            dpo_response = load_model_and_generate(
                dpo_model, scenario["prompt"], scenario["max_tokens"]
            )
            print(f"   {dpo_response}\n")
        except Exception as e:
            print(f"   Error: Model not found. Train DPO first:")
            print(f"   oumi train -c configs/examples/learn_dpo_laptop.yaml\n")

        # Test GRPO Model
        if i == 2:  # Letter counting test
            print("\n🟣 Base Model (Qwen 0.5B - before GRPO):")
            try:
                base_response = load_model_and_generate(
                    base_qwen, scenario["prompt"], scenario["max_tokens"]
                )
                print(f"   {base_response}\n")
            except Exception as e:
                print(f"   Error loading base model: {e}\n")

        print("🟡 GRPO Model (after reward-based training):")
        try:
            grpo_response = load_model_and_generate(
                grpo_model, scenario["prompt"], scenario["max_tokens"]
            )
            print(f"   {grpo_response}\n")
        except Exception as e:
            print(f"   Error: Model not found. Train GRPO first:")
            print(f"   oumi train -c configs/examples/learn_grpo_laptop.yaml\n")

        print_separator("-")
        input("\nPress Enter to continue to next test...")

    # Summary
    print_section("📊 Summary & Insights")

    print("Key Observations:\n")

    print("1. DPO (Preference-based Training):")
    print("   ✓ Better at general conversation quality")
    print("   ✓ More helpful and structured responses")
    print("   ✓ Good for: chat, writing, helpfulness\n")

    print("2. GRPO (Reward-based Training):")
    print("   ✓ Better at specific verifiable tasks")
    print("   ✓ Learns to follow formats (e.g., \\boxed{})")
    print("   ✓ Good for: math, code, reasoning\n")

    print("3. When to Use Each:")
    print("   • DPO: You have human preference data (chosen/rejected pairs)")
    print("   • GRPO: You have a reward function (correct answers, test cases)")
    print("   • Both: For production models (SFT → DPO → GRPO pipeline)\n")

    print_separator()
    print("\n✅ Comparison complete!\n")


def interactive_mode():
    """Interactive mode to test both models with custom prompts."""

    print("\n🎮 Interactive Comparison Mode")
    print_separator()

    dpo_model = "output/learn_dpo/checkpoint-50"
    grpo_model = "output/learn_grpo/checkpoint-30"

    print("Enter your prompts to compare both models.")
    print("Type 'quit' to exit.\n")

    while True:
        prompt = input("Your prompt: ").strip()

        if prompt.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 👋")
            break

        if not prompt:
            continue

        print("\n" + "─" * 70)
        print("🟢 DPO Model:")
        try:
            dpo_response = load_model_and_generate(dpo_model, prompt)
            print(f"   {dpo_response}\n")
        except Exception as e:
            print(f"   Error: {e}\n")

        print("🟡 GRPO Model:")
        try:
            grpo_response = load_model_and_generate(grpo_model, prompt)
            print(f"   {grpo_response}\n")
        except Exception as e:
            print(f"   Error: {e}\n")

        print("─" * 70 + "\n")


if __name__ == "__main__":
    print("\nDPO vs GRPO Comparison Tool")
    print("=" * 70)
    print("\nOptions:")
    print("1. Run predefined comparison tests")
    print("2. Interactive mode (custom prompts)")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        compare_models()
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        print("Goodbye! 👋")
        sys.exit(0)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
