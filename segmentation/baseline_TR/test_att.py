import torch
import torch.nn.functional as F

S, T, B, D = 1024, 1024, 8, 512
source_feats = torch.randn(S, B, D)       # [1024, 8, 512]
target_feats = torch.randn(T, B, D)       # [1024, 8, 512]

# Reshape target_feats to [T*B, D]
target_feats_flat = target_feats.permute(1, 0, 2).reshape(T * B, D)  # [8192, 512]

# Compute attention scores manually
# source_feats: [S, B, D]
# target_feats_flat: [8192, D]

# Compute QK^T manually: [S, B, D] x [D, 8192] → [S, B, 8192]
attn_scores = torch.einsum('sbd,td->sbt', source_feats, target_feats_flat)  # [1024, 8, 8192]

# Softmax over the last dimension (pairing with all target samples)
attn_weights = F.softmax(attn_scores, dim=-1)  # [1024, 8, 8192]

# Now compute output: weights x values
# target_feats_flat: [8192, D]
attn_output = torch.einsum('sbt,td->sbd', attn_weights, target_feats_flat)  # [1024, 8, 512]

print(attn_output.shape)  # Should be [1024, 8, 512]
