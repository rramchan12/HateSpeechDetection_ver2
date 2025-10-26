#!/usr/bin/env python3
"""
Check if LoRA training has converged by analyzing trainer_state.json
"""
import json
import sys
from pathlib import Path


def check_convergence(checkpoint_dir: str, threshold: float = 0.01):
    """
    Analyze if model has converged based on validation loss trends
    
    Args:
        checkpoint_dir: Path to checkpoint directory containing trainer_state.json
        threshold: Minimum improvement percentage to consider as "still improving"
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Find trainer_state.json (either in root or latest checkpoint)
    state_file = checkpoint_path / "trainer_state.json"
    if not state_file.exists():
        # Look for latest checkpoint subdirectory
        checkpoints = sorted(checkpoint_path.glob("checkpoint-*"))
        if checkpoints:
            state_file = checkpoints[-1] / "trainer_state.json"
    
    if not state_file.exists():
        print(f"❌ Error: trainer_state.json not found in {checkpoint_dir}")
        print("   Make sure training has completed at least one epoch.")
        return None
    
    # Load state
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    # Extract eval losses
    eval_logs = [log for log in state['log_history'] if 'eval_loss' in log]
    
    if len(eval_logs) == 0:
        print("❌ No evaluation logs found. Training might not have completed.")
        return None
    
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Total steps: {state['global_step']}")
    print(f"Epochs completed: {state['epoch']}")
    print()
    
    # Display eval losses per epoch
    print("Validation Loss Progression:")
    print("-" * 50)
    for i, log in enumerate(eval_logs, 1):
        epoch = log['epoch']
        loss = log['eval_loss']
        step = log['step']
        print(f"  Epoch {epoch:.1f} (step {step:3d}): eval_loss = {loss:.4f}")
    print()
    
    # Convergence check
    if len(eval_logs) < 2:
        print("⚠️  INSUFFICIENT DATA")
        print("   Only 1 epoch completed. Need at least 2-3 epochs to assess convergence.")
        print("   Recommendation: Continue training for more epochs.")
        return False
    
    # Calculate improvement from last two epochs
    last_loss = eval_logs[-1]['eval_loss']
    prev_loss = eval_logs[-2]['eval_loss']
    improvement = prev_loss - last_loss
    improvement_pct = (improvement / prev_loss) * 100 if prev_loss > 0 else 0
    
    print("Last Two Epochs:")
    print(f"  Previous: {prev_loss:.4f}")
    print(f"  Current:  {last_loss:.4f}")
    print(f"  Change:   {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print()
    
    # Determine convergence
    is_converged = abs(improvement_pct) < threshold * 100
    
    if is_converged:
        print("✅ MODEL LIKELY CONVERGED")
        print(f"   Improvement < {threshold*100}% between last two epochs")
        print("   Recommendation: Training can be stopped.")
    elif improvement > 0:
        print("⏳ MODEL STILL IMPROVING")
        print(f"   Improvement = {improvement_pct:.2f}% (threshold: {threshold*100}%)")
        print("   Recommendation: Continue training for more epochs.")
    else:
        print("⚠️  MODEL PERFORMANCE DEGRADING")
        print("   Validation loss is increasing (possible overfitting)")
        print("   Recommendation: Stop training and use earlier checkpoint.")
    
    print("="*70)
    print()
    
    return is_converged


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_convergence.py <checkpoint_dir> [threshold]")
        print()
        print("Examples:")
        print("  python check_convergence.py finetuning/models/lora_checkpoints")
        print("  python check_convergence.py finetuning/models/lora_checkpoints 0.02")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    
    check_convergence(checkpoint_dir, threshold)


if __name__ == "__main__":
    main()
