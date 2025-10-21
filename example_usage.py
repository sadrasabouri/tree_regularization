#!/usr/bin/env python3
"""
Example script showing how to use TreeReg with your own code.
This demonstrates the three simple steps mentioned in the README.
"""

import torch
import torch.nn as nn
from src.regularizer.regularizer_main import TreeRegularizer

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():  # only works on macOS
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"PyTorch version: {torch.__version__} on {DEVICE}")

def example_with_dummy_data():
    """
    Example showing how to integrate TreeReg into your training loop.
    """
    print("TreeReg Integration Example")
    print("=" * 50)
    
    # Step 1: Initialize TreeRegularizer
    print("Step 1: Initializing TreeRegularizer...")
    regularizer = TreeRegularizer()
    print("✓ TreeRegularizer initialized")
    
    # Step 2: Prepare your data
    print("\nStep 2: Preparing example data...")
    
    # Example: 2 sentences with 4 words each
    batch_size = 2
    seq_len = 4
    hidden_dim = 64
    
    # Simulate hidden states from your model (e.g., from an intermediate layer)
    # These should be the hidden states you want to regularize
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"✓ Created hidden states: {hidden_states.shape}")
    
    # Word boundaries: boolean mask indicating where words start
    # For this example, every token starts a word
    word_boundaries = [
        torch.tensor([True, True, True, True]),  # Sentence 1: 4 words
        torch.tensor([True, True, True, True])   # Sentence 2: 4 words
    ]
    print("✓ Created word boundaries")
    
    # Parse trees: dictionary mapping spans to split points
    # Format: {"start end": split_point} where split_point is 1-indexed
    parses = [
        {
            "0 4": 2,  # Split sentence 1 at position 2 (middle)
            "0 2": 1,  # Split first half at position 1
            "2 4": 3   # Split second half at position 3
            #          ____|____
            #         |         |
            #    0    1    2    3    4
        },
        {
            "0 4": 2,  # Split sentence 2 at position 2 (middle)
            "0 2": 1,  # Split first half at position 1
            "2 4": 3   # Split second half at position 3
            #          ____|____
            #         |         |
            #    0    1    2    3    4
        }
    ]
    print("✓ Created parse trees")
    
    # Step 3: Compute TreeReg loss
    print("\nStep 3: Computing TreeReg loss...")
    
    # Build SCIN charts (this computes the orthogonal scores)
    scin_charts = regularizer.build_chart(hidden_states, word_boundaries, parses)
    print(f"✓ Built SCIN charts: {len(scin_charts)} charts")
    
    # Get TreeReg scores
    device = torch.device(DEVICE)  # Use appropriate device
    treereg_scores, _ = regularizer.get_score(scin_charts, word_boundaries, parses, device)
    print(f"✓ Computed TreeReg scores: {len(treereg_scores)} scores")
    
    # Compute final TreeReg loss
    treereg_loss = -torch.mean(torch.stack(treereg_scores))
    print(f"✓ TreeReg loss: {treereg_loss.item():.4f}")
    
    return treereg_loss

def example_integration_pattern():
    """
    Show how TreeReg would typically be integrated into a training loop.
    """
    print("\n" + "=" * 50)
    print("Integration Pattern Example")
    print("=" * 50)
    
    # Initialize TreeReg
    regularizer = TreeRegularizer()
    
    # Simulate a training batch
    batch_size = 1
    seq_len = 6
    hidden_dim = 128
    
    # Your model's hidden states (e.g., from layer 12 as mentioned in README)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Word boundaries for the sentence
    word_boundaries = [torch.tensor([True, True, True, True, True, True])]
    
    # Parse tree for the sentence
    parse = {
        "0 6": 3,  # Split at position 3
        "0 3": 2,  # Split first part at position 2
        "3 6": 5   # Split second part at position 5
    }
    parses = [parse]
    
    # In your training loop, you would:
    print("In your training loop:")
    print("1. Forward pass through your model")
    print("2. Extract hidden states from desired layer (e.g., layer 12)")
    print("3. Compute TreeReg loss:")
    
    # Compute TreeReg loss
    scin_charts = regularizer.build_chart(hidden_states, word_boundaries, parses)
    treereg_scores, _ = regularizer.get_score(scin_charts, word_boundaries, parses, torch.device(DEVICE))
    treereg_loss = -torch.mean(torch.stack(treereg_scores))
    
    print(f"   treereg_loss = {treereg_loss.item():.4f}")
    
    print("4. Combine with your main loss:")
    print("   total_loss = language_model_loss + treereg_weight * treereg_loss")
    print("5. Backward pass and optimization")

def main():
    """Run the examples."""
    try:
        # Run the basic example
        example_with_dummy_data()
        
        # Show integration pattern
        example_integration_pattern()
        
        print("\n" + "=" * 50)
        print("🎉 Examples completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your dataset with parsed sentences")
        print("2. Integrate TreeReg into your training loop")
        print("3. Use the training commands from the README")
        print("\nFor training, see the commands in the README.md file.")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
