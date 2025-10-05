#!/bin/bash
# Quick-start script for learning DPO and GRPO
# Run this to automatically train and test both methods

set -e  # Exit on error

echo "🎓 DPO vs GRPO Learning Pipeline"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Ask user what they want to do
echo "What would you like to do?"
echo "1) Full pipeline (DPO → GRPO → Compare)"
echo "2) DPO only"
echo "3) GRPO only"
echo "4) Compare existing models"
echo "5) View training logs"
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        MODE="full"
        ;;
    2)
        MODE="dpo"
        ;;
    3)
        MODE="grpo"
        ;;
    4)
        MODE="compare"
        ;;
    5)
        MODE="logs"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# ============================================================================
# PART 1: DPO Training
# ============================================================================

if [[ "$MODE" == "full" ]] || [[ "$MODE" == "dpo" ]]; then
    print_section "📖 PART 1: DPO Training (Preference Learning)"

    echo -e "${YELLOW}About DPO:${NC}"
    echo "- Learns from human preferences (chosen vs rejected responses)"
    echo "- Good for: chat quality, helpfulness, style"
    echo "- Training time: ~10-15 minutes on laptop"
    echo ""

    read -p "Press Enter to start DPO training..."

    echo -e "${GREEN}Starting DPO training...${NC}"
    oumi train -c configs/examples/learn_dpo_laptop.yaml

    echo ""
    echo -e "${GREEN}✓ DPO training complete!${NC}"
    echo "Model saved to: output/learn_dpo/"
    echo ""

    # Quick test
    echo -e "${YELLOW}Quick test - generating a sample response...${NC}"
    echo "Prompt: 'Explain quantum computing simply'"
    echo ""

    # Note: Interactive mode won't work in script, so we skip this
    echo "To test interactively, run:"
    echo "  oumi infer --model.model_name output/learn_dpo/checkpoint-50 --interactive"
    echo ""
fi

# ============================================================================
# PART 2: GRPO Training
# ============================================================================

if [[ "$MODE" == "full" ]] || [[ "$MODE" == "grpo" ]]; then
    print_section "📖 PART 2: GRPO Training (Reward-based Learning)"

    echo -e "${YELLOW}About GRPO:${NC}"
    echo "- Learns from reward functions (correct answers)"
    echo "- Good for: math, coding, reasoning tasks"
    echo "- Training time: ~15-20 minutes on laptop"
    echo ""

    read -p "Press Enter to start GRPO training..."

    echo -e "${GREEN}Starting GRPO training...${NC}"
    oumi train -c configs/examples/learn_grpo_laptop.yaml

    echo ""
    echo -e "${GREEN}✓ GRPO training complete!${NC}"
    echo "Model saved to: output/learn_grpo/"
    echo ""

    # Quick test
    echo -e "${YELLOW}Quick test - letter counting task...${NC}"
    echo "To test interactively, run:"
    echo "  oumi infer --model.model_name output/learn_grpo/checkpoint-30 --interactive"
    echo ""
fi

# ============================================================================
# PART 3: Comparison
# ============================================================================

if [[ "$MODE" == "full" ]] || [[ "$MODE" == "compare" ]]; then
    print_section "📊 PART 3: Comparing DPO vs GRPO"

    echo -e "${YELLOW}Key Differences:${NC}"
    echo ""
    echo "┌─────────────────────┬─────────────────────┬─────────────────────┐"
    echo "│ Aspect              │ DPO                 │ GRPO                │"
    echo "├─────────────────────┼─────────────────────┼─────────────────────┤"
    echo "│ Data Type           │ Preference pairs    │ Reward functions    │"
    echo "│ Training Speed      │ Faster              │ Slower              │"
    echo "│ Memory Usage        │ Lower               │ Higher              │"
    echo "│ Best For            │ Style, helpfulness  │ Math, reasoning     │"
    echo "│ Evaluation          │ Subjective          │ Objective           │"
    echo "└─────────────────────┴─────────────────────┴─────────────────────┘"
    echo ""

    echo -e "${YELLOW}Testing scenarios:${NC}"
    echo ""
    echo "1. Conversation task (DPO's strength):"
    echo "   Prompt: 'Explain why the sky is blue'"
    echo "   Expected: DPO model gives clearer, more helpful response"
    echo ""
    echo "2. Letter counting task (GRPO's strength):"
    echo "   Prompt: 'How many t's in constitution?'"
    echo "   Expected: GRPO model gives correct answer: \\boxed{3}"
    echo ""

    echo -e "${GREEN}To compare both models:${NC}"
    echo ""
    echo "# Test DPO model:"
    echo "oumi infer --model.model_name output/learn_dpo/checkpoint-50 --interactive"
    echo ""
    echo "# Test GRPO model:"
    echo "oumi infer --model.model_name output/learn_grpo/checkpoint-30 --interactive"
    echo ""
fi

# ============================================================================
# View Logs
# ============================================================================

if [[ "$MODE" == "logs" ]]; then
    print_section "📋 Training Logs"

    echo "Available training runs:"
    echo ""

    if [ -d "output/learn_dpo" ]; then
        echo -e "${GREEN}✓ DPO training found${NC}"
        echo "  Location: output/learn_dpo/"
        if [ -f "output/learn_dpo/trainer_state.json" ]; then
            echo "  Status: Complete"
        fi
    else
        echo -e "${RED}✗ DPO training not found${NC}"
    fi

    echo ""

    if [ -d "output/learn_grpo" ]; then
        echo -e "${GREEN}✓ GRPO training found${NC}"
        echo "  Location: output/learn_grpo/"
        if [ -f "output/learn_grpo/trainer_state.json" ]; then
            echo "  Status: Complete"
        fi
    else
        echo -e "${RED}✗ GRPO training not found${NC}"
    fi

    echo ""
    echo "To view detailed logs:"
    echo "  cat output/learn_dpo/training.log    # DPO logs"
    echo "  cat output/learn_grpo/training.log   # GRPO logs"
fi

# ============================================================================
# Final Summary
# ============================================================================

print_section "🎉 Summary"

echo -e "${GREEN}You've learned:${NC}"
echo "✓ How DPO works with preference data"
echo "✓ How GRPO works with reward functions"
echo "✓ When to use each method"
echo "✓ How to train and evaluate both"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "1. Read the detailed guide: cat LEARNING_GUIDE_DPO_GRPO.md"
echo "2. Experiment with your own data"
echo "3. Try larger models (1B → 3B → 7B)"
echo "4. Combine methods (SFT → DPO → GRPO)"
echo ""

echo -e "${BLUE}Happy learning! 🚀${NC}"
