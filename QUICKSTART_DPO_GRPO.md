# 🚀 DPO vs GRPO Quick Start Guide

**Goal**: Learn DPO and GRPO through hands-on training on your laptop.

---

## 📁 Files Created for You

```
oumi/
├── LEARNING_GUIDE_DPO_GRPO.md          # Comprehensive learning guide
├── QUICKSTART_DPO_GRPO.md              # This file (quick reference)
├── learn_dpo_grpo.sh                   # Automated training script
├── compare_dpo_grpo.py                 # Model comparison tool
└── configs/examples/
    ├── learn_dpo_laptop.yaml           # DPO training config (laptop-optimized)
    ├── learn_grpo_laptop.yaml          # GRPO training config (laptop-optimized)
    ├── eval_dpo_results.yaml           # DPO evaluation config
    └── eval_grpo_results.yaml          # GRPO evaluation config
```

---

## ⚡ 3-Minute Quick Start

### Option 1: Automated (Easiest)

```bash
# Run the automated learning script
./learn_dpo_grpo.sh

# Choose: 1 (Full pipeline)
# Wait 30-40 minutes total
# Done! You've trained both DPO and GRPO models
```

### Option 2: Step-by-Step (More Control)

```bash
# 1. Train DPO (10-15 mins)
oumi train -c configs/examples/learn_dpo_laptop.yaml

# 2. Train GRPO (15-20 mins)
oumi train -c configs/examples/learn_grpo_laptop.yaml

# 3. Compare results
python compare_dpo_grpo.py
```

---

## 🎯 What Each Method Does

### DPO (Direct Preference Optimization)
```
Input:  Preference pairs (chosen vs rejected responses)
Output: Model aligned to human preferences
Use:    Chat quality, helpfulness, style

Example:
  Prompt:  "Explain quantum computing"
  Chosen:  "Quantum computing uses quantum bits..."  ✓
  Rejected: "idk, something with atoms"              ✗
```

### GRPO (Group Relative Policy Optimization)
```
Input:  Prompts + Reward function
Output: Model optimized for task accuracy
Use:    Math, code, reasoning tasks

Example:
  Prompt:  "Count 't' in 'constitution'"
  Model:   "\\boxed{3}"
  Reward:  1.0 (correct!)  ✓
```

---

## 📊 Comparison Table

| Feature | DPO | GRPO |
|---------|-----|------|
| **Training Time** | 10-15 min | 15-20 min |
| **Memory Use** | Low (~4GB) | Medium (~6GB) |
| **Data Type** | Preference pairs | Prompts + rewards |
| **Strength** | Conversation | Reasoning |
| **Model Size** | 0.5B (Falcon) | 0.5B (Qwen) |

---

## 🧪 Testing Your Models

### Test DPO Model
```bash
# Interactive chat
oumi infer \
  --model.model_name output/learn_dpo/checkpoint-50 \
  --interactive

# Try: "Explain machine learning simply"
```

### Test GRPO Model
```bash
# Interactive chat
oumi infer \
  --model.model_name output/learn_grpo/checkpoint-30 \
  --interactive

# Try: "How many 'e's in 'Tennessee'?"
# Expected: \\boxed{4}
```

### Side-by-Side Comparison
```bash
python compare_dpo_grpo.py
# Choose option 1 for predefined tests
# Choose option 2 for interactive mode
```

---

## 📈 Expected Results

### DPO Model (After Training)
**Before DPO**: Responses may be terse, less helpful
**After DPO**: More structured, detailed, helpful responses

### GRPO Model (After Training)
**Before GRPO**: May not format answers correctly
**After GRPO**: Correctly formats answers as `\boxed{number}`, higher accuracy

---

## 🔧 Customization

### Make DPO Faster
```yaml
# Edit: configs/examples/learn_dpo_laptop.yaml
training:
  max_steps: 25  # Reduce from 50
  per_device_train_batch_size: 2  # Increase if you have memory
```

### Make GRPO More Accurate
```yaml
# Edit: configs/examples/learn_grpo_laptop.yaml
training:
  max_steps: 50  # Increase from 30
  grpo:
    num_generations: 4  # Increase from 2
```

### Use Your Own Data

**DPO**: Replace dataset in `learn_dpo_laptop.yaml`
```yaml
data:
  train:
    datasets:
      - dataset_name: "your-username/your-dpo-dataset"
```

**GRPO**: Replace dataset + reward function
```yaml
data:
  train:
    datasets:
      - dataset_name: "your-username/your-task-dataset"

training:
  reward_functions: ["your_custom_reward"]  # Define in code
```

---

## 🐛 Troubleshooting

### "Out of Memory" Error

**For DPO:**
```yaml
training:
  per_device_train_batch_size: 1  # Reduce to 1
peft:
  lora_r: 4  # Reduce from 8
```

**For GRPO:**
```yaml
training:
  grpo:
    vllm_gpu_memory_utilization: 0.2  # Reduce from 0.3
    num_generations: 1  # Reduce from 2
```

### "Model not found" Error
```bash
# Make sure training completed
ls output/learn_dpo/    # Should see checkpoint-50
ls output/learn_grpo/   # Should see checkpoint-30

# If missing, check training logs
cat output/learn_dpo/training.log
```

### "vLLM not working" (GRPO)
```yaml
# Disable vLLM, use slower but more compatible mode
training:
  grpo:
    use_vllm: False
```

---

## 📚 Learn More

### Read the Full Guide
```bash
cat LEARNING_GUIDE_DPO_GRPO.md
```

### View Training Progress
```bash
# DPO logs
tail -f output/learn_dpo/training.log

# GRPO logs
tail -f output/learn_grpo/training.log
```

### Explore Other Examples
```bash
# See all Oumi recipes
ls configs/recipes/

# SmolLM (even smaller models)
ls configs/recipes/smollm*/

# Other DPO examples
find configs/recipes -name "*dpo*"
```

---

## 🎓 Next Steps

1. **✅ Complete this quickstart** (~30-40 mins)
2. **📖 Read the full learning guide** (`LEARNING_GUIDE_DPO_GRPO.md`)
3. **🔬 Try different prompts** (use `compare_dpo_grpo.py`)
4. **📊 Compare with base models** (see before/after improvements)
5. **🚀 Scale up**:
   - Try 1B or 3B models (if you have GPU memory)
   - Use more training data
   - Combine: SFT → DPO → GRPO pipeline

---

## 💡 Key Takeaways

✅ **DPO = Preference Learning**
- Need: Human-labeled preference pairs
- Result: Better conversation quality
- Use when: You want style/helpfulness improvements

✅ **GRPO = Reward Learning**
- Need: Reward function (correct answers)
- Result: Better task accuracy
- Use when: You have verifiable tasks (math, code)

✅ **Both are valuable!**
- Production models often use both
- DPO for general quality, GRPO for specific skills
- Start with whichever matches your data

---

## ⏱️ Time Budget

- **Setup**: 2 minutes
- **DPO Training**: 10-15 minutes
- **GRPO Training**: 15-20 minutes
- **Testing & Comparison**: 5-10 minutes
- **Total**: ~30-50 minutes

---

## 🎉 You're Ready!

Run this to start:
```bash
./learn_dpo_grpo.sh
```

Or jump straight to training:
```bash
# DPO first
oumi train -c configs/examples/learn_dpo_laptop.yaml

# Then GRPO
oumi train -c configs/examples/learn_grpo_laptop.yaml
```

Good luck! 🚀
