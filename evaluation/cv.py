from evaluation.evaluation import ComparativeEvaluation

import torch
import torch.nn.functional as F

class CCV(ComparativeEvaluation):
    def evaluate(self, nat_xs, nat_ys, *_):
        prob_nat, pred_nat = self.model(nat_xs).softmax(1).max(1)
        prob_aug, pred_aug = self.model_aug(nat_xs).softmax(1).max(1)
        diff = correct = 0
        for i in range(len(nat_ys)):
            if nat_ys[i] == pred_nat[i] == pred_aug[i]:
                correct += 1
                diff += (prob_aug[i] - prob_nat[i]).abs_().item()
        diff /= correct
        return {
            "Classification Confidence Variance": diff,
            "Percentage of misclassifications": 1-correct/len(nat_ys)
        }

class COS(ComparativeEvaluation):
    def evaluate(self, nat_xs, nat_ys, *_):
        prob_nat, pred_nat= self.model(nat_xs).softmax(1).max(1)
        prob_aug, pred_aug = self.model_aug(nat_xs).softmax(1).max(1)
        div = correct = 0
        for i in range(len(nat_ys)):
            if nat_ys[i] == pred_nat[i] == pred_aug[i]:
                correct += 1
                m = ((prob_nat[i]+prob_aug[i])/2).log_()
                div += F.kl_div(m, prob_nat[i], reduction="batchmean") + F.kl_div(m,
                    prob_aug[i], reduction="batchmean")
        div /= 2 * correct
        return {
            "Classification Output Stability": div.item(),
            "Percentage of misclassifications": 1-correct/len(nat_ys)
        }
