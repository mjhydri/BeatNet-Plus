# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
#
# BeatNet+ neural network models.
# Implements:
#   - BeatNetPlusBranch: Single CRNN branch (Conv1d + 4-layer LSTM + Linear)
#   - BeatNetPlus: Dual-branch model for generic training (main + auxiliary)
#   - AuxiliaryFreezing: Teacher-student adaptation (frozen pre-trained teacher)
#   - GuidedFineTuning: Single-branch fine-tuning (wraps BeatNetPlusBranch)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BeatNetPlusBranch(nn.Module):
    """Single CRNN branch for BeatNet+.

    Architecture: Conv1d(1,2,k=10) -> ReLU -> MaxPool1d(2) -> Linear -> 4-layer LSTM -> Linear -> [3 classes]

    This is the core building block. During inference, a single branch is used.
    During training, two branches can be combined (BeatNetPlus, AuxiliaryFreezing).
    """

    def __init__(self, dim_in=288, num_cells=150, num_layers=4, device='cpu'):
        super().__init__()
        self.dim_in = dim_in
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.device = device

        self.kernel_size = 10
        self.conv_out = num_cells

        # Convolutional block (identical to BeatNet)
        self.conv1 = nn.Conv1d(1, 2, self.kernel_size)
        conv_flat_dim = 2 * ((dim_in - self.kernel_size + 1) // 2)
        self.linear0 = nn.Linear(conv_flat_dim, self.conv_out)

        # Recurrent block (4 layers instead of BeatNet's 2)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=num_cells,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Output head
        self.output_linear = nn.Linear(num_cells, 3)

        # Hidden state for streaming inference
        self.hidden = torch.zeros(num_layers, 1, num_cells).to(device)
        self.cell = torch.zeros(num_layers, 1, num_cells).to(device)

        self.to(device)

    def _extract_features(self, data):
        """Conv + Linear feature extraction (shared by all forward modes)."""
        B, T, D = data.shape
        x = data.reshape(-1, D).unsqueeze(1)           # (B*T, 1, dim_in)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)     # (B*T, 2, conv_dim)
        x = x.view(x.size(0), -1)                      # (B*T, flat)
        x = self.linear0(x)                             # (B*T, conv_out)
        x = x.reshape(B, T, self.conv_out)              # (B, T, conv_out)
        return x

    def forward(self, data):
        """Stateful forward for streaming inference. Maintains LSTM hidden state."""
        x = self._extract_features(data)
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        out = self.output_linear(x).transpose(1, 2)    # (B, 3, T)
        return out

    def train_forward(self, data):
        """Stateless forward for training. Returns both logits and latent embedding."""
        x = self._extract_features(data)               # (B, T, conv_out)
        x = self.lstm(x)[0]                             # (B, T, num_cells)
        latent = self.output_linear(x)                  # (B, T, 3) — latent embedding
        logits = latent.transpose(1, 2)                 # (B, 3, T)
        return logits, latent

    def inference_forward(self, data):
        """Stateless forward for batch inference. Returns logits only."""
        logits, _ = self.train_forward(data)
        return logits

    def reset_hidden(self):
        """Reset LSTM hidden state (call before processing a new audio stream)."""
        self.hidden = torch.zeros(self.num_layers, 1, self.num_cells).to(self.device)
        self.cell = torch.zeros(self.num_layers, 1, self.num_cells).to(self.device)

    @staticmethod
    def softmax_pred(logits):
        """Apply softmax to logits for inference."""
        return F.softmax(logits, dim=1)  # softmax over class dimension


class BeatNetPlus(nn.Module):
    """Dual-branch BeatNet+ for generic training.

    Main branch: receives full music mixture.
    Auxiliary branch: receives non-percussive (drumless) version of the same piece.
    Connected by MSE latent-matching loss during training.
    Only the main branch is used at inference.

    Paper: Section 3.1.2, Equation 1.
    """

    def __init__(self, dim_in=288, num_cells=150, num_layers=4, device='cpu'):
        super().__init__()
        self.main_branch = BeatNetPlusBranch(dim_in, num_cells, num_layers, device)
        self.aux_branch = BeatNetPlusBranch(dim_in, num_cells, num_layers, device)
        self.device = device

    def train_forward(self, main_input, aux_input):
        """Forward both branches. Returns logits and latents for loss computation.

        Returns
        -------
        main_logits : (B, 3, T) — main branch logits
        aux_logits : (B, 3, T) — auxiliary branch logits
        main_latent : (B, T, 3) — main branch latent embedding
        aux_latent : (B, T, 3) — auxiliary branch latent embedding
        """
        main_logits, main_latent = self.main_branch.train_forward(main_input)
        aux_logits, aux_latent = self.aux_branch.train_forward(aux_input)
        return main_logits, aux_logits, main_latent, aux_latent

    def inference_forward(self, data):
        """Inference using main branch only."""
        return self.main_branch.inference_forward(data)

    def get_main_state_dict(self):
        """Get state dict of main branch only (for saving/inference)."""
        return self.main_branch.state_dict()

    @staticmethod
    def compute_loss(main_logits, aux_logits, main_latent, aux_latent,
                     targets, class_weights, mse_lambda=200.0):
        """Compute total BeatNet+ loss (Equation 1 in paper).

        L_total = L_CE_main + L_CE_aux + mse_lambda * L_MSE(latent_main, latent_aux)
        """
        ce_main = F.cross_entropy(main_logits, targets, weight=class_weights)
        ce_aux = F.cross_entropy(aux_logits, targets, weight=class_weights)
        mse = F.mse_loss(main_latent, aux_latent)
        total = ce_main + ce_aux + mse_lambda * mse
        return total, {'ce_main': ce_main.item(), 'ce_aux': ce_aux.item(),
                        'mse': mse.item(), 'total': total.item()}


class AuxiliaryFreezing(nn.Module):
    """Auxiliary Freezing (AF) adaptation model.

    Frozen teacher: pre-trained BeatNet+ main branch, receives full music mixture.
    Student: randomly initialized branch, trained on target domain (vocals, non-percussive).
    Connected by MSE latent-matching loss.
    Only the student branch is used at inference after adaptation.

    Paper: Section 3.2, Figure 2.
    """

    def __init__(self, dim_in=288, num_cells=150, num_layers=4, device='cpu',
                 pretrained_weights=None):
        super().__init__()
        # Student branch (randomly initialized, will be trained)
        self.student = BeatNetPlusBranch(dim_in, num_cells, num_layers, device)
        # Teacher branch (frozen, loaded from pre-trained weights)
        self.teacher = BeatNetPlusBranch(dim_in, num_cells, num_layers, device)

        if pretrained_weights is not None:
            self.teacher.load_state_dict(
                torch.load(pretrained_weights, map_location=device), strict=False
            )
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.device = device

    def train_forward(self, student_input, teacher_input):
        """Forward both branches.

        Parameters
        ----------
        student_input : (B, T, dim_in) — target domain audio features
        teacher_input : (B, T, dim_in) — full mix audio features

        Returns
        -------
        student_logits, teacher_logits, student_latent, teacher_latent
        """
        student_logits, student_latent = self.student.train_forward(student_input)
        with torch.no_grad():
            teacher_logits, teacher_latent = self.teacher.train_forward(teacher_input)
        return student_logits, teacher_logits, student_latent, teacher_latent

    def inference_forward(self, data):
        """Inference using student branch only."""
        return self.student.inference_forward(data)

    def get_student_state_dict(self):
        """Get state dict of student branch (for saving/inference)."""
        return self.student.state_dict()

    @staticmethod
    def compute_loss(student_logits, student_latent, teacher_latent,
                     targets, class_weights, mse_lambda=200.0):
        """AF loss: CE on student + MSE between student and frozen teacher latents."""
        ce = F.cross_entropy(student_logits, targets, weight=class_weights)
        mse = F.mse_loss(student_latent, teacher_latent)
        total = ce + mse_lambda * mse
        return total, {'ce': ce.item(), 'mse': mse.item(), 'total': total.item()}


class GuidedFineTuning(nn.Module):
    """Guided Fine-Tuning (GF) adaptation wrapper.

    Single-branch model initialized from pre-trained BeatNet+ weights.
    Fine-tuned with gradual data adaptation: accompaniment intensity is linearly
    reduced each epoch by decay_rate.

    Paper: Section 3.3, Figure 3.
    """

    def __init__(self, dim_in=288, num_cells=150, num_layers=4, device='cpu',
                 pretrained_weights=None):
        super().__init__()
        self.branch = BeatNetPlusBranch(dim_in, num_cells, num_layers, device)

        if pretrained_weights is not None:
            self.branch.load_state_dict(
                torch.load(pretrained_weights, map_location=device), strict=False
            )

        self.device = device

    def train_forward(self, data):
        """Standard single-branch forward. The data scheduling (mixing target + fading
        accompaniment) is handled by the dataset, not the model."""
        logits, latent = self.branch.train_forward(data)
        return logits

    def inference_forward(self, data):
        return self.branch.inference_forward(data)

    def get_state_dict(self):
        return self.branch.state_dict()

    @staticmethod
    def compute_loss(logits, targets, class_weights):
        """Standard CE loss for GF (no MSE needed — adaptation is in the data)."""
        ce = F.cross_entropy(logits, targets, weight=class_weights)
        return ce, {'ce': ce.item(), 'total': ce.item()}