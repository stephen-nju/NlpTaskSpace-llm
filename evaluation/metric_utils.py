# ner 测评方案

import collections
import json
import math
import re
import string
from typing import List

from tqdm import tqdm


def has_duplicate(tmp_list):
    """has duplicate ?"""
    if tmp_list == []:
        return False

    if type(tmp_list[0]) == str:
        if len(tmp_list) == len(set(tmp_list)):
            return False
        else:
            return True

    if type(tmp_list[0]) == list:
        tmp = []
        for t in tmp_list:
            if t not in tmp:
                tmp.append(t)
        if len(tmp_list) == len(tmp):
            return False
        else:
            return True


def get_correct_list_from_response_list(target_list, response_list):
    """
    target_list 和 response_list 均有可能包含重复的 item
    """
    # print(f"{target_list}=={response_list}")
    res = []
    if not has_duplicate(response_list):
        res = [item for item in response_list if item in target_list]
    else:
        if not has_duplicate(target_list):
            # 去重
            uni_response_list = []
            for item in response_list:
                if item not in uni_response_list:
                    uni_response_list.append(item)
            res = [item for item in uni_response_list if item in target_list]
        else:
            res = []
            processed_item_list = []
            for item in response_list:
                if item not in processed_item_list:
                    processed_item_list.append(item)

                    num_item = response_list.count(item)
                    if num_item == 1:  # not duplicate
                        if item in target_list:
                            res.append(item)
                    else:  # duplicate
                        if item in target_list:
                            num_item_in_target = target_list.count(item)
                            num_item_correct = min([num_item, num_item_in_target])
                            res += [item] * num_item_correct

    return res


def print_metrics(tp, fp, fn, task):
    p, r, f1 = 0.0, 0.0, 0.0

    if tp + fp != 0:
        p = 1.0 * tp / (tp + fp)
    if tp + fn != 0:
        r = 1.0 * tp / (tp + fn)
    if p + r != 0.0:
        f1 = 2.0 * p * r / (p + r)

    print(
        "{}\n| p: {:.4f}, r: {:.4f}, f1: {:.4f} | tp: {:4d}, fp: {:4d}, fn: {:4d}, tp+fn: {:4d}\n".format(
            task,
            round(p, 4),
            round(r, 4),
            round(f1, 4),
            tp,
            fp,
            fn,
            tp + fn,
        )
    )
    return p, r, f1


## report overall metric
def report_metric(inputs):
    data = inputs["data"]
    total = inputs["total"]
    # e_types_list = ["HP", "HC", "XL"]
    ## per type
    e_types_list = []
    for example in data:
        ground_truth = example["ground_truth"]
        for truth in ground_truth:
            e_types_list.append(truth["type"])

    e_types_list = list(set(e_types_list))

    hard_boundaries = dict()
    for key in e_types_list:
        hard_boundaries[key] = {"tp": 0, "fp": 0, "fn": 0}

    num_entity = 0
    tp_ner_boundaries = 0
    fp_ner_boundaries = 0
    fn_ner_boundaries = 0
    tp_ner_strict = 0
    fp_ner_strict = 0
    fn_ner_strict = 0

    for example in tqdm(data, total=total):
        ## target
        strict_target_list = []
        boundaries_target_list = []

        ## predict
        strict_predict_list = []
        boundaries_predict_list = []
        ## per type target
        boundaries_target_list_dict = {}
        # per type predict
        boundaries_predict_list_dict = {}

        for key in e_types_list:
            boundaries_target_list_dict[key] = []
            boundaries_predict_list_dict[key] = []

        ground_truth = example["ground_truth"]
        for truth in ground_truth:
            # print(ground_truth)
            type_name = truth["type"]
            if type_name in e_types_list:
                # start = truth["start"]
                # end = truth["end"]
                span = truth["span"].lower()
                # 总类型
                # 统一转化为小写字母进行比较
                strict_target_list.append([type_name, span])
                boundaries_target_list.append(span)
                ## per type
                boundaries_target_list_dict[type_name].append(span)
                num_entity += 1

        predict = example["baichuan"]
        for ent in predict:
            ent_name = ent["span"].lower()
            ent_type = ent["type"]
            if ent_type in e_types_list:
                strict_predict_list.append([ent_type, ent_name])
                boundaries_predict_list.append(ent_name)
                # per type
                boundaries_predict_list_dict[ent_type].append(ent_name)

            ## hard-match
        strict_correct_list = get_correct_list_from_response_list(strict_target_list, strict_predict_list)
        # boundaries_correct_list = get_correct_list_from_response_list(boundaries_target_list, boundaries_predict_list)

        tp_ner_strict += len(strict_correct_list)
        fp_ner_strict += len(strict_predict_list) - len(strict_correct_list)
        fn_ner_strict += len(strict_target_list) - len(strict_correct_list)

        # tp_ner_boundaries += len(boundaries_correct_list)
        # fp_ner_boundaries += len(boundaries_predict_list) - len(boundaries_correct_list)
        # fn_ner_boundaries += len(boundaries_target_list) - len(boundaries_correct_list)

        for key in e_types_list:
            cur_correct = get_correct_list_from_response_list(
                boundaries_target_list_dict[key], boundaries_predict_list_dict[key]
            )
            hard_boundaries[key]["tp"] += len(cur_correct)
            hard_boundaries[key]["fp"] += len(boundaries_predict_list_dict[key]) - len(cur_correct)
            hard_boundaries[key]["fn"] += len(boundaries_target_list_dict[key]) - len(cur_correct)

    print("#sentence: {}, #entity: {}".format(len(data), num_entity))

    p, r, f1 = print_metrics(tp_ner_strict, fp_ner_strict, fn_ner_strict, "NER-strict-hardMatch")

    # per type
    for key in e_types_list:
        print_metrics(
            hard_boundaries[key]["tp"],
            hard_boundaries[key]["fp"],
            hard_boundaries[key]["fn"],
            f"Ner-strict-hardmatch-{key}",
        )

    return f1
