"""
LISA-Gemma3 モデルの損失関数モジュール
"""

import torch
import torch.nn.functional as F


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """
    DICE損失を計算します（マスク用のIOU損失に類似）
    Args:
        inputs: 任意の形状の浮動小数点テンソル
                各サンプルの予測値
        targets: inputsと同じ形状の浮動小数点テンソル
                各要素のバイナリ分類ラベルを格納
                (0: ネガティブクラス、1: ポジティブクラス)
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    シグモイドクロスエントロピー損失を計算します
    Args:
        inputs: 任意の形状の浮動小数点テンソル
                各サンプルの予測値
        targets: inputsと同じ形状の浮動小数点テンソル
                各要素のバイナリ分類ラベルを格納
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss


class DiceLoss(torch.nn.Module):
    """
    DICE損失のモジュール実装
    """
    def __init__(self, scale=1000, eps=1e-6):
        super().__init__()
        self.scale = scale
        self.eps = eps
        
    def forward(self, inputs, targets, num_masks=1, reduction="mean"):
        """
        DICE損失を計算する
        
        Args:
            inputs (torch.Tensor): 予測マスク [B, num_masks, H, W]
            targets (torch.Tensor): 正解マスク [B, num_masks, H, W]
            num_masks (int): マスクの数
            reduction (str): 損失値の縮約方法 ('mean'または'sum')
            
        Returns:
            torch.Tensor: 損失値
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        
        numerator = 2 * torch.sum(inputs * targets, dim=-1)
        denominator = torch.sum(inputs, dim=-1) + torch.sum(targets, dim=-1) + self.eps
        
        loss = 1 - (numerator / denominator)
        
        if reduction == "none":
            return loss
        
        # バッチとマスクの平均を取る
        if num_masks == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        return loss.sum() / (loss.shape[0] * num_masks) if reduction == "mean" else loss.sum()


class SigmoidCELoss(torch.nn.Module):
    """
    シグモイドクロスエントロピー損失のモジュール実装
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets, num_masks=1, reduction="mean"):
        """
        シグモイドクロスエントロピー損失関数
        
        Args:
            inputs (torch.Tensor): 予測マスク [B, num_masks, H, W]
            targets (torch.Tensor): 正解マスク [B, num_masks, H, W]
            num_masks (int): マスクの数
            reduction (str): 損失値の縮約方法 ('mean'または'sum')
            
        Returns:
            torch.Tensor: 損失値
        """
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        ).mean(dim=-1)
        
        if reduction == "none":
            return loss
        
        # バッチとマスクの平均を取る
        if num_masks == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        return loss.sum() / (loss.shape[0] * num_masks) if reduction == "mean" else loss.sum()


class CompositeLoss(torch.nn.Module):
    """
    LISA-Gemma3用の複合損失関数
    テキスト生成損失 + セグメンテーション損失
    """
    def __init__(self, ce_loss_weight=1.0, dice_loss_weight=0.5, bce_loss_weight=2.0):
        super().__init__()
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = SigmoidCELoss()
        
    def forward(self, outputs, batch):
        """
        複合損失を計算する
        
        Args:
            outputs: モデルの出力辞書
            batch: バッチデータ
            
        Returns:
            dict: 各損失値を含む辞書
        """
        losses = {}
        total_loss = 0.0
        
        # テキスト生成損失（Gemma-3のlanguage modeling loss）
        if "gemma_logits" in outputs and "labels" in batch:
            labels = batch["labels"]
            logits = outputs["gemma_logits"]
            
            # シフトしてlanguage modeling lossを計算
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # -100でマスクされたトークンは無視
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            text_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            losses["text_loss"] = text_loss
            total_loss += self.ce_loss_weight * text_loss
        
        # セグメンテーション損失
        if "predicted_masks" in outputs and outputs["predicted_masks"] is not None:
            predicted_masks = outputs["predicted_masks"]
            
            if "ground_truth_mask" in batch and batch["ground_truth_mask"] is not None:
                ground_truth_mask = batch["ground_truth_mask"]
                
                # マスクの数を計算
                num_masks = predicted_masks.size(0) if predicted_masks.dim() > 2 else 1
                
                # DICE損失
                dice_loss_val = self.dice_loss(predicted_masks, ground_truth_mask, num_masks)
                losses["dice_loss"] = dice_loss_val
                total_loss += self.dice_loss_weight * dice_loss_val
                
                # BCE損失
                bce_loss_val = self.bce_loss(predicted_masks, ground_truth_mask, num_masks)
                losses["bce_loss"] = bce_loss_val
                total_loss += self.bce_loss_weight * bce_loss_val
        
        losses["total_loss"] = total_loss
        return losses 