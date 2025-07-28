import pytest
import torch

def test_hrm():
    from HRM.hrm import HRM

    hrm = HRM(
        networks = [
            dict(
                dim = 512,
                depth = 2,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            ),
            dict(
                dim = 512,
                depth = 4,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            )
        ],
        num_tokens = 256,
        dim = 512,

    )

    seq = torch.randint(0, 256, (3, 1024))
    labels = torch.randint(0, 256, (3, 1024))

    loss, (logits, hiddens) = hrm(seq, labels = labels)
    loss.backward()
