import json
import re
import argparse


def modify_keys_in_json(input_file, output_file):
    # JSON 파일 로드
    with open(input_file, 'r') as f:
        data = json.load(f)

    modified_data = {}

    # 각 키에 대해 변환 작업 수행
    for key in data.keys():
        # 띄어쓰기 제거
        new_key = key.replace(" ", "")
        # 'coco' 부분을 'COCO'로 변경
        new_key = re.sub(r'coco', 'COCO', new_key, flags=re.IGNORECASE)

        # 수정된 키와 원래 값으로 새로운 데이터 저장
        modified_data[new_key] = data[key]

    # 변환된 데이터를 새로운 JSON 파일로 저장
    with open(output_file, 'w') as f:
        json.dump(modified_data, f, indent=4)

    input_name = input_file.rsplit('/', 1)[-1]
    output_name = output_file.rsplit('/', 1)[-1]
    print("Finished modifying %s to %s" % (input_name, output_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-candidate', type=str,
                        default='/mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal-PC/seed-1234/candidate-5Q.json',
                        help='Path for train annotation json file.')
    parser.add_argument('--output-candidate', type=str,
                        default='/mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal-PC/seed-1234/candidate-5Q-modified.json',
                        help='Path for train annotation json file.')

    parser.add_argument('--input-reference', type=str,
                        default='/mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal-PC/seed-1234/reference-5Q.json',
                        help='Path for train annotation json file.')
    parser.add_argument('--output-reference', type=str,
                        default='/mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal-PC/seed-1234/reference-5Q-modified.json',
                        help='Path for train annotation json file.')

    args = parser.parse_args()

    modify_keys_in_json(args.input_candidate, args.output_candidate)
    modify_keys_in_json(args.input_reference, args.output_reference)

# # 사용 예시
# input_file = 'input.json'  # 원래 JSON 파일 경로
# output_file = 'output.json'  # 변환된 JSON 파일을 저장할 경로
# modify_keys_in_json(input_file, output_file)
