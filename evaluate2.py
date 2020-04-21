"""Evaluation script

Note that this evaluation only cares about the phrase level extraction.
Also, it does not care where the phrases come from.

Usage:
    python3 evaluate.py thai_ner_dev_set_answers.json my_answer.json
"""
import json
import sys
import pandas as pd

def evaluate_answer(gold_file, prediction_file):
    gold = json.load(open(gold_file))
    pred = json.load(open(prediction_file))
    assert set(gold.keys()) == set(pred.keys()), 'check filenames!'
    ent_types = ['place', 'org', 'pers']
    total_correct, total_pred, total_gold = 0, 0, 0  # for micro
    df = pd.DataFrame(columns=['PRECISION','RECALL','F1','SUPPORT'])
    pd.options.display.float_format = '{:.4f}'.format
    for ent_type in ent_types:
        type_correct, type_pred, type_gold = 0, 0, 0  # for macro
        for corpus_file in gold:
            gold_ner_set = set(gold[corpus_file][ent_type])
            pred_ner_set = set(pred[corpus_file][ent_type])
            total_pred += len(pred_ner_set)
            total_gold += len(gold_ner_set)
            total_correct += len(gold_ner_set.intersection(pred_ner_set))
            type_pred += len(pred_ner_set)
            type_gold += len(gold_ner_set)
            type_correct += len(gold_ner_set.intersection(pred_ner_set))
        precision = type_correct / type_pred
        recall = type_correct / type_gold
        f1 = (2 * precision * recall) / (precision + recall)
        df.loc[ent_type] = [precision, recall, f1, str(type_gold)]
    precision_micro = total_correct / total_pred
    recall_micro = total_correct / total_gold
    f1_micro = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro)
    df.loc['MACRO'] = [df.PRECISION.mean(), df.RECALL.mean(), df.F1.mean(), str(total_gold)]
    df.loc['MICRO'] = [precision_micro, recall_micro, f1_micro, str(total_gold)]
    print(df)

if __name__ == '__main__':
    gold_file = sys.argv[1]
    prediction_file = sys.argv[2]
    evaluate_answer(gold_file, prediction_file)
