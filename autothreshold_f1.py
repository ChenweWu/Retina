import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import precision_recall_curve, f1
from typing import Optional

class AutoThresholdF1(Metric):
  def __init__(self, num_classes: int, average: Optional[str] = None, 
               compute_on_step=False, dist_sync_on_step=False):
    super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

    self.num_classes = num_classes if num_classes else 1

    if average and average != "macro":
      raise NotImplementedError(f"Reduction {average} not implemented.")
    self.average = average

    self.thresholds = torch.full((self.num_classes,), 0.5)
    self.update_thresholds = True
    self.add_state("preds", default=[])
    self.add_state("target", default=[])
  
  def update(self, preds: Tensor, target: Tensor):
    self.preds.append(preds)
    self.target.append(target)
  
  def freeze(self):
    self.update_thresholds = False
  
  def unfreeze(self):
    self.update_thresholds = True

  def compute(self):
    preds = torch.cat(self.preds, dim=0)
    target = torch.cat(self.target, dim=0)
    output = torch.zeros(self.num_classes)
    
    if self.update_thresholds:
      for i in range(self.num_classes):
        try:
          p, r, t = precision_recall_curve(preds[..., i], target[..., i])

          f1_scores = 2 * (p * r) / (p + r)
          f1_scores.masked_fill_(f1_scores.isnan(), 0)

          max_f1_idx = f1_scores.argmax()
          self.thresholds[i] = t[max_f1_idx]
          output[i] = f1_scores[max_f1_idx]
        except Exception as e:
          print(f"Could not calculate F1 for class {i}: {e}")
    else:
      for i in range(self.num_classes):
        try:
          output[i] = f1(preds[..., i], target[..., i], threshold=self.thresholds[i])
        except Exception as e:
          print(f"Could not calculate F1 for class {i}: {e}")

    return output.mean() if self.average else output
  
  @property
  def is_differentiable(self):
    return False