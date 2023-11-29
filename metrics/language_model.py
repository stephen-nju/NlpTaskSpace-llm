import jieba
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from sklearn.metrics import accuracy_score
from torchmetrics import Metric


class LanguageModelMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", [])
        self.add_state("labels", [])

    def update(self, pred, label):
        # print(f"pred shape={pred.shape},target shape={label.shape}")
        self.preds.append(pred)
        self.labels.append(label)

    def compute(self, tokenizer, ignore_pad_token_for_loss, global_rank):
        # print(f"leN({self.preds})===self.preds shape={self.preds[0].shape}==self.preds={self.preds[0]}")

        for preds, labels in zip(self.preds, self.labels):
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
            for pred, label in zip(decoded_preds, decoded_labels):
                hypothesis = list(jieba.cut(pred))
                reference = list(jieba.cut(label))
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            for k, v in score_dict.items():
                score_dict[k] = float(np.mean(v))
            return score_dict


class AccMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", [])
        self.add_state("labels", [])

    def update(self, pred, label):
        # print(f"pred shape={pred.shape},target shape={label.shape}")
        self.preds.append(pred)
        self.labels.append(label)

    def compute(self, global_rank, normalize=True, sample_weight=None):
        p, l = [], []
        for preds, labels in zip(self.preds, self.labels):
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()
            p.extend(preds)
            l.extend(labels)
        acc = float(accuracy_score(l, p, normalize=normalize, sample_weight=sample_weight))
        if global_rank == 0:
            print(acc)

        return acc
