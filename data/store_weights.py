import h5py
import numpy as np
import json
import argparse
import os

def read_answer_types_from_h5py(file_path):
    with h5py.File(file_path, 'r') as f:
        answer_types = f['answer_types'][:]
    return answer_types

def calculate_class_sample_counts(labels):
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_classes, class_counts))

def calculate_sample_weights(labels, class_sample_counts):
    weights = 1./np.array([class_sample_counts[label] for label in labels], dtype=np.float32)
    return weights

def save_weights_to_json(weights, file_path):
    with open(file_path, 'w') as f:
        json.dump(weights.tolist(), f)

def create_weights_from_h5py(args):
    file_path = os.path.join(args.file_path, 'iq_val_dataset.hdf5')
    output_json_path = os.path.join(args.output_json_path, 'iq_val_dataset_weights.json')

    answer_types = read_answer_types_from_h5py(file_path)
    class_sample_counts = calculate_class_sample_counts(answer_types)
    sample_weights = calculate_sample_weights(answer_types, class_sample_counts)
    save_weights_to_json(sample_weights, output_json_path)

    return sample_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file-path', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split',
                        help='file path where h5py exists')
    parser.add_argument('--output-json-path', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split',
                        help='path to save dataset weights file')
    args = parser.parse_args()

    weights = create_weights_from_h5py(args)
