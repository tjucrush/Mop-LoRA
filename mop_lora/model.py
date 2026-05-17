import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MoPLoRALayer(nn.Module):
    """
    MoP-LoRA Core Layer: Mixture of Paths LoRA
    Introduces multi-path A matrices and SiLU-gated non-linear activations
    to break the linear bottleneck of traditional LoRA.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, num_paths: int = 4, lora_alpha: int = 16, dropout_p: float = 0.05):
        super().__init__()
        self.r = r
        self.num_paths = num_paths
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        # Multi-path A matrices (Mixture of Paths)
        # Each path has its own low-rank projection
        self.lora_A_paths = nn.ParameterList([
            nn.Parameter(torch.empty((r, in_features))) for _ in range(num_paths)
        ])
        
        # Single B matrix to project back to out_features
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))

        # Gating network for dynamic path routing
        self.gating = nn.Linear(in_features, num_paths, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize B matrix to zeros so that initial state is identity-like to base weights
        nn.init.zeros_(self.lora_B)
        
        # Kaiming uniform for A matrices
        for A in self.lora_A_paths:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            
        # Initialize gating to zero for equal path contribution at start
        nn.init.zeros_(self.gating.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoP-LoRA
        x: Input tensor of shape [..., in_features]
        """
        # 1. Calculate routing weights for paths
        # gates shape: [..., num_paths]
        gates = torch.softmax(self.gating(x), dim=-1)

        # 2. Compute non-linear multi-path activations
        path_outputs = []
        for i, A in enumerate(self.lora_A_paths):
            # Linear projection A_i
            A_out = F.linear(x, A)
            # SiLU non-linear activation
            A_out = F.silu(A_out)
            # Gate weight integration
            path_outputs.append(A_out * gates[..., i:i+1])

        # Sum selected paths
        combined_A = sum(path_outputs)
        combined_A = self.dropout(combined_A)

        # 3. Final B projection and scaling
        out = F.linear(combined_A, self.lora_B) * self.scaling
        return out


class MoPLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear incorporating MoP-LoRA.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, num_paths: int = 4, lora_alpha: int = 16, dropout_p: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        self.mop_lora = MoPLoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            r=r,
            num_paths=num_paths,
            lora_alpha=lora_alpha,
            dropout_p=dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate base output (frozen) + MoP-LoRA update
        base_out = self.base_layer(x)
        lora_out = self.mop_lora(x)
        return base_out + lora_out
