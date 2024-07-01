import torch
import torch.linalg as LA
import torchvision.transforms.v2.functional as F

from evaluation.evaluation import Evaluation


class ALDp(Evaluation):
    """
    ALDp is the average normalized Lp distortion of successful samples.
    """

    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        # Scale to RGB values
        nat_xs = F.to_dtype(nat_xs, dtype=torch.uint8, scale=True)
        adv_xs = F.to_dtype(adv_xs, dtype=torch.uint8, scale=True)
        # adv_xs = v2.ToDtype(adv_xs, scale=True)
        # nat_xs = v2.ToDtype(nat_xs, scale=True)

        pert = adv_xs - nat_xs

        correct = l0_dist = l2_dist = linf_dist = 0
        for i in range(len(adv_out)):
            if adv_out[i].argmax(1) != nat_ys[i]:
                correct += 1
                l0_dist += (
                    LA.vector_norm(pert[i], ord="0").item()
                    / LA.vector_norm(nat_xs[i], ord="0").item()
                )
                l2_dist += (
                    LA.vector_norm(pert[i], ord="2").item()
                    / LA.vector_norm(nat_xs[i], ord="2").item()
                )
                linf_dist += (
                    LA.vector_norm(pert[i], ord="inf").item()
                    / LA.vector_norm(nat_xs[i], ord="inf").item()
                )
                assert LA.vector_norm(pert[i]).item() == LA.vector_norm(
                    torch.reshape(pert[i], -1)
                )

        return {
            "Average L0 distortion": l0_dist / correct,
            "Average L2 distortion": l2_dist / correct,
            "Average Linf distortion": linf_dist / correct,
            "Percentage of successful attacks": correct / len(nat_xs),
        }
