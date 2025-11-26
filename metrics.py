from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json

def tokenize(refs, cands, no_op=False):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {idx: [{'caption':r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption':c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands

def bleu(gts, res):
    #gts, res = tokenize(gts, res)
    scorer = Bleu(n=4) # n=4: bleu-1부터 4까지 보겠다
    # score: 전체 데이터셋에 대한 점수 리스트 (bleu-1, 2, 3, 4 포함)
    # scores: 각 데이터에 대한 점수 리스트 (bleu-1, 2, 3, 4 포함)
    score, scores = scorer.compute_score(gts, res)
    # ratio: 생성된 문장과 레퍼런스 문장 간의 길이 비율, BLEU 점수에 페널티 적용 여부 결정 (1보다 작을 경우 페널티 부여)
    return score


def calculate_self_bleu_original(sentences):
    """
    Calculate Self-BLEU score for a list of sentences using pycocoevalcap.

    Parameters:
    sentences (list of str): List of sentences to evaluate

    Returns:
    tuple: Self-BLEU scores for BLEU-1, BLEU-2, BLEU-3, BLEU-4
    """
    num_sentences = len(sentences)
    total_scores = [0, 0, 0, 0]

    for i in range(num_sentences):
        references = {i: []}
        candidate = {i: [sentences[i]]}

        for j in range(num_sentences):
            if i != j:
                references[i].append(sentences[j])

        score = bleu(references, candidate)
        for k in range(4):
            total_scores[k] += score[k]

    self_bleu_scores = [total_score / num_sentences for total_score in total_scores]
    return self_bleu_scores

def calculate_self_bleu(sentences):
    num_sentences = len(sentences)
    total_scores = [0, 0, 0, 0]

    for i in range(num_sentences):
        references = []
        candidate = []
        candidate.append(sentences[i])

        for j in range(num_sentences):
            if i != j:
                references.append(sentences[j])

        references = [references]
        references, candidate = tokenize(references, candidate)
        score = bleu(references, candidate)
        for k in range(4):
            total_scores[k] += score[k]

    self_bleu_scores = [total_score / num_sentences for total_score in total_scores]
    return self_bleu_scores


def calculate_self_cider(sentences):
    """
    Calculate Self-CIDEr score for a list of sentences using pycocoevalcap.

    Parameters:
    sentences (list of str): List of sentences to evaluate

    Returns:
    float: Self-CIDER score
    """
    num_sentences = len(sentences)
    total_score = 0.0

    # Tokenize sentences using PTBTokenizer
    #sentences = tokenize_sentences(sentences)

    for i in range(num_sentences):
        references = {i: [sentences[j] for j in range(num_sentences) if i != j]}
        candidate = {i: [sentences[i]]}

        score = cider(references, candidate)
        total_score += score

    self_cider_score = total_score / num_sentences
    return self_cider_score



def cider(gts, res):
    #gts, res = tokenize(gts, res)
    scorer = Cider()
    # score: 전체 데이터셋에 대한 점수
    # scores: 각 데이터에 대한 점수 리스트
    (score, scores) = scorer.compute_score(gts, res)
    # cider: 0부터 1이상의 값까지 나올 수 있음.
    # perfect match: 2 이상 값 가능
    # good match: 1 이상 값 가능
    # bad match: 거의 0에 가까운 값
    return score

def meteor(gts, res):
    #gts, res = tokenize(gts, res)
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    return score

def rouge(gts, res):
    #gts, res = tokenize(gts, res)
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    return score

def spice(gts, res):
    #gts, res = tokenize(gts, res)
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    return score

def main(gts_list, res_list):
    # gts, res = tokenize(gts_list, res_list)
    gts = {i: [caption] for i, caption in enumerate(gts_list)}
    res = {i: [caption] for i, caption in enumerate(res_list)}

    print("gts keys:", gts.keys())
    print("res keys:", res.keys())

    bleu_score = bleu(gts, res) # [bleu_1, bleu_2, bleu_3, bleu_4]
    cider_score = cider(gts, res)
    meteor_score = meteor(gts, res)
    rouge_score = rouge(gts, res)
    # spice()

    return float(bleu_score[0])*100.0, float(bleu_score[1])*100.0, float(bleu_score[2])*100.0, float(bleu_score[3])*100.0, cider_score*100.0, meteor_score*100.0, rouge_score*100.0
