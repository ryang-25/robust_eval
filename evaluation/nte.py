from evaluation.evaluation import Evaluation

class NTE(Evaluation):
    def evaluate(self, nat_xs, nat_ys, adv_xs, adv_out):
        prob, pred = adv_out.softmax(1).topk(2, sorted=True)
        diff = misclass = 0
        for i in range(len(pred)):
            if pred[i, 0] != nat_ys[i]:
                misclass += 1
                adv_prob, next_prob = prob[i]
                diff += (adv_prob - next_prob).item()
        return {"Noise tolerance estimation": diff / len(pred)}
