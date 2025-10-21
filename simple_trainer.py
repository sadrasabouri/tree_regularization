#!/usr/bin/env python3
"""
Simple TreeReg Trainer
A simplified trainer based on example_usage.py and the repository structure.
This demonstrates how to train a transformer language model with TreeReg regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import random
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys

# Add src to path
sys.path.append('src')

# Import from the repository
from regularizer.regularizer_main import TreeRegularizer
from models.transformer_lm import TransformerLM
from vocabulary import CharVocabulary
from data.data_utils import get_word_boundaries

class SimpleTreeDataset(Dataset):
    """Simple dataset for demonstration with synthetic parse trees."""
    
    def __init__(self, sentences, parses, vocab, max_len=50):
        self.sentences = sentences
        self.parses = parses
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        parse = self.parses[idx]
        
        # Tokenize sentence
        tokens = self.vocab(sentence)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        # Pad sequence
        length = len(tokens)
        tokens += [0] * (self.max_len - length)  # 0 is pad token
        
        # Get word boundaries
        word_boundaries = get_word_boundaries(sentence, self.vocab, hf=False)
        if len(word_boundaries) > self.max_len:
            word_boundaries = word_boundaries[:self.max_len]
        word_boundaries += [0] * (self.max_len - len(word_boundaries))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'word_boundaries': torch.tensor(word_boundaries, dtype=torch.bool),
            'parses': json.dumps(parse),
            'sentence': sentence
        }

def create_synthetic_data(num_samples=1000):
    """Create synthetic training data with parse trees."""
    sentences = []
    parses = []
    
    # Simple sentence templates with known parse structures
    templates = [
        ("The cat sat on the mat", {"0 6": 3}),  # Split at "sat"
        ("A dog runs quickly", {"0 4": 2}),      # Split at "runs"
        ("She reads books", {"0 3": 1}),         # Split at "reads"
        ("They play games together", {"0 4": 2}), # Split at "play"
        ("I love programming", {"0 3": 1}),       # Split at "love"
        ("The bird flies high", {"0 4": 2}),     # Split at "flies"
        ("We eat dinner", {"0 3": 1}),           # Split at "eat"
        ("Children play outside", {"0 3": 1}),   # Split at "play"
        ("The sun shines bright", {"0 4": 2}),   # Split at "shines"
        ("Students study hard", {"0 3": 1}),     # Split at "study"
    ]
    
    for _ in range(num_samples):
        template, parse = random.choice(templates)
        sentences.append(template)
        parses.append(parse)
    
    return sentences, parses

def collate_fn(batch):
    """Custom collate function for the dataset."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    word_boundaries = [item['word_boundaries'].tolist() for item in batch]
    parses = [item['parses'] for item in batch]
    sentences = [item['sentence'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'length': lengths,
        'word_boundaries': word_boundaries,
        'parses': parses,
        'sentences': sentences
    }

class SimpleTreeRegTrainer:
    """Simple trainer for TreeReg with transformer language models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Create vocabulary
        self.vocab = self._create_vocabulary()
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Create TreeReg regularizer
        self.regularizer = TreeRegularizer(orth_bidir=args.orth_bidir)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_steps, eta_min=args.min_lr
        )
        
        print(f"Initialized trainer on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_vocabulary(self):
        """Create a character-level vocabulary."""
        # Get all unique characters from the dataset
        all_chars = set()
        sentences, _ = create_synthetic_data(100)  # Sample to get chars
        for sentence in sentences:
            all_chars.update(sentence)
        
        # Add special tokens
        all_chars.update(['<pad>', '<eos>', '<sos>'])
        
        vocab = CharVocabulary(chars=all_chars, ignore_char='<pad>', ignore_char_idx=0)
        return vocab
    
    def _create_model(self):
        """Create the transformer language model."""
        model = TransformerLM(
            n_input_tokens=len(self.vocab),
            state_size=self.args.hidden_dim,
            n_heads=self.args.n_heads,
            encoder_n_layers=self.args.n_layers,
            embedding_dropout=self.args.embedding_dropout,
            output_dropout=self.args.output_dropout,
            relative=self.args.relative,
            max_len=self.args.max_len
        )
        return model
    
    def _prepare_batch(self, batch):
        """Prepare batch for model input."""
        input_ids = batch['input_ids'].to(self.device)
        lengths = batch['length'].to(self.device)
        
        # Create target (shifted input for language modeling)
        targets = input_ids.clone()
        targets[:, :-1] = input_ids[:, 1:]
        targets[:, -1] = 0  # Pad last token
        
        return input_ids, lengths, targets
    
    def _compute_lm_loss(self, input_ids, lengths, targets):
        """Compute language modeling loss."""
        # Forward pass
        result = self.model(
            src=input_ids,
            src_len=lengths,
            get_hidden_states=True,
            layer_id=self.args.treereg_layer,
            sci_heads=self.args.sci_heads
        )
        
        # Get logits and hidden states
        logits = result.data
        hidden_states = result.hidden_states
        
        # Compute cross-entropy loss
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return loss, hidden_states
    
    def _compute_treereg_loss(self, hidden_states, word_boundaries, parses):
        """Compute TreeReg regularization loss."""
        # Convert word boundaries to the expected format
        word_boundaries_list = []
        for wb in word_boundaries:
            word_boundaries_list.append(torch.tensor(wb, dtype=torch.bool))
        
        # Convert parses to the expected format
        parses_list = []
        for parse_str in parses:
            parse_dict = json.loads(parse_str)
            parses_list.append(parse_dict)
        
        # Build SCIN charts
        scin_charts = self.regularizer.build_chart(
            hidden_states, word_boundaries_list, parses_list
        )
        
        # Get TreeReg scores
        treereg_scores, _ = self.regularizer.get_score(
            scin_charts, word_boundaries_list, parses_list, self.device
        )
        
        # Compute TreeReg loss
        if treereg_scores:
            treereg_loss = -torch.mean(torch.stack(treereg_scores))
        else:
            treereg_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return treereg_loss
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_treereg_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            input_ids, lengths, targets = self._prepare_batch(batch)
            
            # Forward pass
            lm_loss, hidden_states = self._compute_lm_loss(input_ids, lengths, targets)
            
            # Compute TreeReg loss
            treereg_loss = torch.tensor(0.0, device=self.device)
            if self.args.use_treereg and batch_idx % self.args.treereg_steps == 0:
                treereg_loss = self._compute_treereg_loss(
                    hidden_states, batch['word_boundaries'], batch['parses']
                )
            
            # Total loss
            total_loss_batch = lm_loss + self.args.treereg_weight * treereg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += lm_loss.item()
            total_treereg_loss += treereg_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'LM Loss': f"{lm_loss.item():.4f}",
                'TreeReg Loss': f"{treereg_loss.item():.4f}",
                'Total Loss': f"{total_loss_batch.item():.4f}",
                'LR': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_treereg_loss = total_treereg_loss / num_batches
        
        return avg_loss, avg_treereg_loss
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.args.epochs):
            # Train epoch
            train_loss, train_treereg_loss = self.train_epoch(train_dataloader, epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, TreeReg Loss = {train_treereg_loss:.4f}")
            
            # Validation (if provided)
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(epoch)
    
    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids, lengths, targets = self._prepare_batch(batch)
                lm_loss, _ = self._compute_lm_loss(input_ids, lengths, targets)
                total_loss += lm_loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        os.makedirs(self.args.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple TreeReg Trainer')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--relative', action='store_true', help='Use relative positional encoding')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum training steps')
    
    # TreeReg parameters
    parser.add_argument('--use_treereg', action='store_true', help='Use TreeReg regularization')
    parser.add_argument('--treereg_weight', type=float, default=0.1, help='TreeReg loss weight')
    parser.add_argument('--treereg_steps', type=int, default=1, help='Apply TreeReg every N steps')
    parser.add_argument('--treereg_layer', type=int, default=-1, help='Layer to apply TreeReg')
    parser.add_argument('--sci_heads', type=float, default=0.5, help='Fraction of heads for SCI')
    parser.add_argument('--orth_bidir', action='store_true', help='Use bidirectional orthogonalization')
    
    # Other parameters
    parser.add_argument('--embedding_dropout', type=float, default=0.1, help='Embedding dropout')
    parser.add_argument('--output_dropout', type=float, default=0.1, help='Output dropout')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--save_every', type=int, default=5, help='Save every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SimpleTreeRegTrainer(args)
    
    # Create datasets
    train_sentences, train_parses = create_synthetic_data(1000)
    val_sentences, val_parses = create_synthetic_data(200)
    
    train_dataset = SimpleTreeDataset(train_sentences, train_parses, trainer.vocab, args.max_len)
    val_dataset = SimpleTreeDataset(val_sentences, val_parses, trainer.vocab, args.max_len)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Train
    trainer.train(train_dataloader, val_dataloader)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
