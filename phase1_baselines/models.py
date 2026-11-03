"""
models.py — Per-Modality Neural Network Architectures
=====================================================
Four small-but-reasonable models (100K–500K params each):
  - ImageCNN:  Conv2D-based classifier for 32×32 RGB images
  - TextLSTM:  Embedding + LSTM classifier for token-ID sequences
  - AudioCNN:  Conv2D classifier for MFCC-shaped spectrograms
  - VideoCNN:  Frame-level Conv2D + temporal pooling for video clips

All models output class logits (no softmax — use CrossEntropyLoss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Image CNN ───────────────────────────────────────────────────────────────

class ImageCNN(nn.Module):
    """
    Small CNN for 32×32 RGB images.
    Architecture: Conv(3→32)→BN→ReLU→Pool → Conv(32→64)→BN→ReLU→Pool
                  → Conv(64→128)→BN→ReLU→Pool → FC(128*4*4→256)→FC(256→num_classes)
    ~340K params for 10 classes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # (B,32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # (B,32,16,16)

            nn.Conv2d(32, 64, 3, padding=1),   # (B,64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # (B,64,8,8)

            nn.Conv2d(64, 128, 3, padding=1),  # (B,128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # (B,128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Text LSTM ───────────────────────────────────────────────────────────────

class TextLSTM(nn.Module):
    """
    Embedding + bidirectional LSTM for text classification.
    Architecture: Embedding(vocab→64) → BiLSTM(64→128) → FC(256→128) → FC(128→num_classes)
    ~350K params for 4 classes, vocab=5000.
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64,
                 hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 128),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len) of token IDs
        emb = self.embedding(x)               # (B, seq_len, embed_dim)
        output, (h_n, _) = self.lstm(emb)     # h_n: (2, B, hidden_dim)
        # Concatenate final hidden states from both directions
        hidden = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, hidden_dim*2)
        return self.classifier(hidden)


# ── Audio CNN ───────────────────────────────────────────────────────────────

class AudioCNN(nn.Module):
    """
    CNN for MFCC-shaped spectrograms (1, 40, 100).
    Architecture: Conv(1→32)→BN→ReLU→Pool → Conv(32→64)→BN→ReLU→Pool
                  → Conv(64→128)→BN→ReLU→AdaptivePool(4,4)
                  → FC(128*4*4→256)→FC(256→num_classes)
    ~300K params for 10 classes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),    # (B,32,40,100)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # (B,32,20,50)

            nn.Conv2d(32, 64, 3, padding=1),   # (B,64,20,50)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # (B,64,10,25)

            nn.Conv2d(64, 128, 3, padding=1),  # (B,128,10,25)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),       # (B,128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Video CNN ───────────────────────────────────────────────────────────────

class VideoCNN(nn.Module):
    """
    Frame-level 2D CNN + temporal average pooling for video clips.
    Input: (B, T, C, H, W) = (B, 8, 3, 32, 32).
    Each frame is processed by a shared CNN, then frame features are averaged.
    Architecture: SharedConv → TemporalAvgPool → FC(128→64) → FC(64→num_classes)
    ~350K params for 5 classes.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Per-frame feature extractor (shared across frames)
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    # (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # (32,16,16)

            nn.Conv2d(32, 64, 3, padding=1),   # (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # (64,8,8)

            nn.Conv2d(64, 128, 3, padding=1),  # (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),       # (128,2,2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Process all frames at once: reshape to (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        features = self.frame_cnn(x)            # (B*T, 128, 2, 2)
        features = features.view(B, T, -1)      # (B, T, 512)
        # Temporal average pooling across frames
        features = features.mean(dim=1)          # (B, 512)
        return self.classifier(features)


# ── Helper Functions ────────────────────────────────────────────────────────

def get_model(modality: str) -> nn.Module:
    """Factory function to get the correct model for a modality."""
    models = {
        "image": lambda: ImageCNN(num_classes=10),
        "text":  lambda: TextLSTM(vocab_size=5000, num_classes=4),
        "audio": lambda: AudioCNN(num_classes=10),
        "video": lambda: VideoCNN(num_classes=5),
    }
    if modality not in models:
        raise ValueError(f"Unknown modality: {modality}. Choose from {list(models.keys())}")
    return models[modality]()


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Print architecture summaries
    print(f"\n{'='*60}")
    print("  MODEL ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    for modality in ["image", "text", "audio", "video"]:
        model = get_model(modality)
        n_params = count_parameters(model)
        print(f"\n  {modality.upper()} — {model.__class__.__name__}")
        print(f"    Parameters: {n_params:,}")

        # Quick forward pass test
        if modality == "image":
            dummy = torch.randn(2, 3, 32, 32)
        elif modality == "text":
            dummy = torch.randint(0, 5000, (2, 50))
        elif modality == "audio":
            dummy = torch.randn(2, 1, 40, 100)
        elif modality == "video":
            dummy = torch.randn(2, 8, 3, 32, 32)

        out = model(dummy)
        print(f"    Input:  {tuple(dummy.shape)}")
        print(f"    Output: {tuple(out.shape)}")
    print(f"\n{'='*60}\n")
