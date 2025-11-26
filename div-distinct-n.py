from paddlenlp.metrics import Distinct
import argparse
import json
import re


def clean_text(text):
    text = text.lower()                     # 소문자화
    text = re.sub(r'[^\w\s]', '', text)     # 구두점 제거
    text = re.sub(r'\s+', ' ', text).strip()# 여백 정리
    return text

def calc_distinct(candidates):
    """
    candidates: List[List[str]] 형태
    """

    # 전체 문장(flatten)
    all_sentences = [clean_text(sent) for group in candidates for sent in group]
    unique_sentences = list(set(all_sentences))

    results = {
        "All": {},
        "Unique": {}
    }

    for n in [1, 2]:
        # 전체 문장 기준
        distinct_all = Distinct(n)
        for sent in all_sentences:
            distinct_all.add_inst(sent.split())
        results["All"][f"Distinct-{n}"] = distinct_all.score()

        # 고유 문장 기준
        distinct_unique = Distinct(n)
        for sent in unique_sentences:
            distinct_unique.add_inst(sent.split())
        results["Unique"][f"Distinct-{n}"] = distinct_unique.score()

    return results


def print_distinct_results(results):
    print("✅ Distinct Scores")
    for mode in ["All", "Unique"]:
        print(f"\n▶ Based on {'All sentences' if mode == 'All' else 'Unique sentences only'}:")
        for k, v in results[mode].items():
            print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True,
                        help='Path to JSON file with format {id: [q1, q2, ..., q5]}')
    args = parser.parse_args()

    # Load JSON
    with open(args.candidate, 'r') as f:
        data = json.load(f)

    candidates = list(data.values())  # List of List[str]

    results = calc_distinct(candidates)
    print_distinct_results(results)
