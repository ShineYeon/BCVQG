from metrics import *
import argparse
import json

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def calcMBLEU(candidate):
    tokenizer = PTBTokenizer()

    total_mBLEU = [0.0, 0.0, 0.0, 0.0]

    num_candidate = len(candidate)

    candidate = {idx: [{'caption': c }for c in c_cands] for idx, c_cands in enumerate(candidate)}
    candidate = tokenizer.tokenize(candidate)
    candidate = list(candidate.values())


    for i, cand in enumerate(candidate):
        mBLEU = calculate_self_bleu_original(cand)

        for j in range(4):
            total_mBLEU[j] += mBLEU[j]


    self_BLEU = {}
    for i in range(4):
        self_BLEU[f"mBLEU-{i+1}"] = total_mBLEU[i] / num_candidate

    print_mBLEU_results(self_BLEU)

    return self_BLEU


def print_mBLEU_results(results):
    print("✅ Distinct Scores - mBLEU")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='./configs/vqg.yaml')

    args = parser.parse_args()

    with open(args.candidate, 'r') as f:
        json_cand = json.load(f)

    candidate = list(json_cand.values())

    calcMBLEU(candidate)
