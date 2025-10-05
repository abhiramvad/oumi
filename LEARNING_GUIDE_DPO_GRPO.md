# 🎓 Hands-On Learning Guide: DPO vs GRPO

This guide will teach you Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) through practical examples you can run on your laptop.

## 📚 Prerequisites

```bash
# Ensure oumi is installed with GPU support
pip install oumi[gpu]

# Or if on Mac with MPS:
pip install oumi
```

## 🎯 Part 1: Train with DPO (Preference Learning)

### What You'll Learn
- How to train models on human preference data
- Understanding chosen vs rejected responses
- When to use DPO (style alignment, safety, helpfulness)

### Step 1: Understand the Data

```bash
# Inspect a DPO dataset
python3 << 'EOF'
from datasets import load_dataset
ds = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train[:5]")

for example in ds:
    print("=" * 50)
    print("PROMPT:", example['prompt'][:100], "...")
    print("\n✅ CHOSEN:", example['chosen'][:100], "...")
    print("\n❌ REJECTED:", example['rejected'][:100], "...")
    print()
EOF
```

**Key Insight**: DPO learns to prefer "chosen" over "rejected" responses.

### Step 2: Train Your DPO Model

```bash
# Train (takes ~10-15 mins on laptop)
oumi train -c configs/examples/learn_dpo_laptop.yaml

# Watch the logs - you'll see:
# - loss decreasing (model learning preferences)
# - rewards increasing for chosen responses
```

### Step 3: Test Your DPO Model

```bash
# Interactive chat with your trained model
oumi infer \
  -c configs/examples/eval_dpo_results.yaml \
  --model.model_name output/learn_dpo/checkpoint-50 \
  --interactive

# Try prompts like:
# - "Explain quantum computing to a 5-year-old"
# - "What's the best way to learn programming?"
```

### Step 4: Compare Before/After

```bash
# Test BEFORE training (base model)
oumi infer \
  --model.model_name "tiiuae/Falcon-H1-0.5B-Instruct" \
  --interactive

# Test AFTER training (your DPO model)
oumi infer \
  --model.model_name output/learn_dpo/checkpoint-50 \
  --interactive
```

**What to Notice**: DPO model should give more helpful, well-structured responses.

---

## 🎯 Part 2: Train with GRPO (Reasoning Tasks)

### What You'll Learn
- How to train on verifiable rewards (correct answers)
- Understanding reward functions
- When to use GRPO (math, code, reasoning tasks)

### Step 1: Understand the Task

```bash
# See letter counting examples
python3 << 'EOF'
from datasets import load_dataset
ds = load_dataset("oumi-ai/oumi-letter-count", split="train[:5]")

for example in ds:
    print("=" * 50)
    print("PROMPT:", example['prompt'][1]['content'])
    print("ANSWER:", example['letter_count'])
    print()
EOF
```

**Key Insight**: GRPO generates multiple answers and learns from a reward function that scores correctness.

### Step 2: Understand the Reward Function

```bash
# See how letter counting is scored
python3 << 'EOF'
# GRPO generates: "The word has \\boxed{3} letter l's"
# Reward function:
# 1. Extracts the number from \\boxed{...}
# 2. Compares to ground truth
# 3. Returns score (1.0 for correct, 0.0 for wrong, partial for format errors)
print("This is automatic - the reward function is built into Oumi!")
EOF
```

### Step 3: Train Your GRPO Model

```bash
# Train (takes ~15-20 mins - more expensive than DPO)
oumi train -c configs/examples/learn_grpo_laptop.yaml

# Watch the logs - you'll see:
# - Multiple generations per prompt (num_generations=2)
# - Reward scores for each generation
# - Model learning which responses get higher rewards
```

### Step 4: Test Your GRPO Model

```bash
# Test on letter counting
oumi infer \
  --model.model_name output/learn_grpo/checkpoint-30 \
  --interactive

# Try:
# - "How many 'l's are in 'hello'?"
# - "Count the letter 'a' in 'banana'"
# - "How many times does 'e' appear in 'Tennessee'?"
```

---

## 📊 Part 3: Compare DPO vs GRPO

### Key Differences

| Aspect | DPO | GRPO |
|--------|-----|------|
| **Data Required** | Preference pairs (chosen/rejected) | Prompts + reward function |
| **Training Speed** | Faster (no generation) | Slower (generates multiple responses) |
| **Best For** | Style, tone, helpfulness | Math, code, reasoning |
| **Memory** | Lower | Higher (needs generation) |
| **Evaluation** | Subjective quality | Objective accuracy |

### Practical Experiment

Create a test script:

```bash
cat > test_both_models.py << 'EOF'
#!/usr/bin/env python3
"""Compare DPO vs GRPO models"""

from oumi import infer

# Test prompts
CONVERSATION_PROMPT = "Explain why the sky is blue in simple terms."
REASONING_PROMPT = "How many times does the letter 't' appear in 'constitution'?"

print("=" * 60)
print("TEST 1: Conversation Quality (DPO's strength)")
print("=" * 60)

# DPO model
print("\n🔵 DPO Model:")
# Your DPO model should do well here

# GRPO model
print("\n🟢 GRPO Model:")
# GRPO model trained on letter counting won't be optimized for this

print("\n" + "=" * 60)
print("TEST 2: Letter Counting (GRPO's strength)")
print("=" * 60)

print("\n🔵 DPO Model:")
# DPO model not trained on reasoning

print("\n🟢 GRPO Model:")
# GRPO model should excel here

print("\n" + "=" * 60)
print("KEY INSIGHT:")
print("- DPO: Better at general conversation & style")
print("- GRPO: Better at specific tasks with clear rewards")
print("=" * 60)
EOF

chmod +x test_both_models.py
```

---

## 🎓 Learning Outcomes

After completing this guide, you'll understand:

✅ **DPO**:
- How preference learning works
- When labeled preference data is valuable
- Training models for subjective quality

✅ **GRPO**:
- How RL with reward functions works
- When verifiable tasks are available
- Training for objective correctness

✅ **Practical Skills**:
- Reading training logs
- Comparing model outputs
- Choosing the right method for your task

---

## 🚀 Next Steps

### 1. **Combine Both** (Advanced)
```bash
# First train with SFT, then DPO, then GRPO
# This is how many production models are built!
```

### 2. **Try Your Own Data**
```bash
# DPO: Create preference pairs for your domain
# GRPO: Write custom reward functions
```

### 3. **Scale Up**
```bash
# Use larger models (1B → 3B → 7B)
# Use more training data
# Try cloud GPUs for faster training
```

### 4. **Read the Research**
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [GRPO/RLOO Paper](https://arxiv.org/abs/2402.14740)
- [Oumi Docs](https://oumi.ai/docs)

---

## 💡 Pro Tips

1. **Start Small**: Use the laptop configs first, understand the concepts
2. **Monitor Logs**: Watch loss and rewards - they tell you if learning is happening
3. **Compare Outputs**: Always test before/after to see what changed
4. **Iterate**: Try different learning rates, batch sizes, num_generations
5. **Know When to Use Each**:
   - DPO: Chat, writing, safety alignment
   - GRPO: Math, code, factual QA, reasoning

---

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce `per_device_train_batch_size` to 1
- Reduce `vllm_gpu_memory_utilization` to 0.2
- Use smaller `lora_r` (try 4)

**Training Too Slow?**
- Reduce `max_steps` further
- Use smaller dataset splits
- Disable gradient checkpointing (uses more memory but faster)

**Models Not Improving?**
- Check learning rate (too high = unstable, too low = no learning)
- Increase `max_steps`
- Verify your data is loading correctly

---

Good luck! 🎉
