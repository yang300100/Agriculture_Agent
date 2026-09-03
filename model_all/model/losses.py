"""HMPD-Net 多任务损失函数。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HMPDLoss(nn.Module):
    """联合分类为主任务，作物/病害/严重度为辅助任务。"""

    def __init__(
        self,
        joint_to_crop,
        joint_to_disease,
        crop_weight=0.3,
        disease_weight=0.3,
        severity_weight=0.2,
        consistency_weight=0.1,
        label_smoothing=0.1,
    ):
        super().__init__()
        self.register_buffer("joint_to_crop", torch.as_tensor(joint_to_crop, dtype=torch.long))
        self.register_buffer(
            "joint_to_disease", torch.as_tensor(joint_to_disease, dtype=torch.long)
        )
        self.crop_weight = crop_weight
        self.disease_weight = disease_weight
        self.severity_weight = severity_weight
        self.consistency_weight = consistency_weight
        self.label_smoothing = label_smoothing

    @staticmethod
    def _marginalize(joint_probs, mapping, output_size):
        marginal = joint_probs.new_zeros((joint_probs.shape[0], output_size))
        indices = mapping.unsqueeze(0).expand(joint_probs.shape[0], -1)
        return marginal.scatter_add(1, indices, joint_probs)

    def forward(self, outputs, targets):
        joint_loss = F.cross_entropy(
            outputs["final_logits"], targets["joint"],
            label_smoothing=self.label_smoothing,
        )
        crop_loss = F.cross_entropy(
            outputs["crop_logits"], targets["crop"],
            label_smoothing=self.label_smoothing,
        )
        disease_loss = F.cross_entropy(
            outputs["disease_logits"], targets["disease"],
            label_smoothing=self.label_smoothing,
        )

        severity_mask = targets["severity"] >= 0
        if severity_mask.any():
            severity_loss = F.cross_entropy(
                outputs["severity_logits"][severity_mask],
                targets["severity"][severity_mask],
                label_smoothing=self.label_smoothing,
            )
        else:
            severity_loss = outputs["severity_logits"].sum() * 0.0

        joint_probs = torch.softmax(outputs["joint_logits"], dim=1)
        crop_marginal = self._marginalize(
            joint_probs, self.joint_to_crop, outputs["crop_logits"].shape[1]
        )
        disease_marginal = self._marginalize(
            joint_probs, self.joint_to_disease, outputs["disease_logits"].shape[1]
        )
        crop_consistency = F.kl_div(
            F.log_softmax(outputs["crop_logits"], dim=1),
            crop_marginal.clamp_min(1e-8),
            reduction="batchmean",
        )
        disease_consistency = F.kl_div(
            F.log_softmax(outputs["disease_logits"], dim=1),
            disease_marginal.clamp_min(1e-8),
            reduction="batchmean",
        )
        consistency_loss = 0.5 * (crop_consistency + disease_consistency)

        total = (
            joint_loss
            + self.crop_weight * crop_loss
            + self.disease_weight * disease_loss
            + self.severity_weight * severity_loss
            + self.consistency_weight * consistency_loss
        )
        return {
            "total": total,
            "joint": joint_loss.detach(),
            "crop": crop_loss.detach(),
            "disease": disease_loss.detach(),
            "severity": severity_loss.detach(),
            "consistency": consistency_loss.detach(),
        }
