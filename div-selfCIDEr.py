import argparse
import json

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider

# PTBTokenizer 사용하여 문장 토큰화
def tokenize_sentences(sentences):
    tokenizer = PTBTokenizer()
    return {idx: tokenizer.tokenize([sentence]) for idx, sentence in enumerate(sentences)}


from pycocoevalcap.cider.cider import Cider


def calculate_self_cider(cand_sentences):
    """
    Calculate Self-CIDEr score for a list of candidate sentences.

    Parameters:
    cand_sentences (list of list of str): List of tokenized sentences for each candidate group

    Returns:
    float: Self-CIDER score for the candidate group
    """
    num_sentences = len(cand_sentences)
    total_score = 0.0

    # Calculate the CIDEr score for each candidate in the list
    for i in range(num_sentences):
        references = {i: [cand_sentences[j] for j in range(num_sentences) if i != j]}  # references = others' candidates
        candidate = {i: [cand_sentences[i]]}  # candidate = current sentence

        score = cider(references, candidate)
        total_score += score

    self_cider_score = total_score / num_sentences
    return self_cider_score


def cider(gts, res):
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    return score


def calcSelfCIDEr(candidate):
    """
    Calculates the Self-CIDEr score for a list of list of generated sentences.

    Parameters:
    candidate (list of list of str): A list where each entry contains 5 candidate sentences.

    Returns:
    dict: A dictionary containing the average Self-CIDEr score.
    """
    total_selfCIDEr = 0.0
    num_candidate = len(candidate)

    # Iterate through each candidate set (list of 5 generated sentences)
    for i, cand_set in enumerate(candidate):
        selfCIDEr = calculate_self_cider(cand_set)  # Calculate self-CIDEr for each set of candidate sentences
        total_selfCIDEr += selfCIDEr

    # Average the scores
    ret_self_CIDEr = {}
    ret_self_CIDEr["self-CIDEr"] = total_selfCIDEr / num_candidate
    print("Self-CIDEr Results:", ret_self_CIDEr)

    return ret_self_CIDEr


def print_selfCIDER_results(results):
    print("✅ Distinct Scores - selfCIDEr")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='./configs/vqg.yaml')

    args = parser.parse_args()

    with open(args.candidate, 'r') as f:
        json_cand = json.load(f)

    candidate = list(json_cand.values())

    calcSelfCIDEr(candidate)
