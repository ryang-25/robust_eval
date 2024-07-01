from evaluation.evaluation import Evaluation

class ACAC(Evaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob, pred = adv_out.softmax(1).max(1)
        misclass = total_prob = 0
        for i in range(len(pred)):
            if pred[i] != nat_ys[i]:
                misclass += 1
                total_prob += prob[i]
        return {
            "Average Confidence of Adversarial Class": total_prob / misclass,
            "Percentage of misclassifications": misclass / len(pred),
        }


class ACTC(Evaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob = adv_out.softmax(1)
        pred = adv_out.argmax(1)
        misclass = total_prob = 0
        for i in range(len(pred)):
            if pred[i] != nat_ys[i]:
                misclass += 1
                total_prob += prob[i, nat_ys[i]]
        return {
            "Average Confidence of True Class": total_prob / misclass,
            "Percentage of misclassifications": misclass / len(pred),
        }
