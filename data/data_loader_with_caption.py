"""Loads question answering and image captioning data and feeds it to the models.
"""

import h5py
import json
from collections import defaultdict
import numpy as np
from models.blip import init_tokenizer
from data.randaugment import RandomAugment

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class VQGWithCaptionDataset(Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, vqg_path, caption_path, tokenizer, transform=None, max_examples=None, indices=None):
        """Initialize the dataset with VQG and image caption data.

        Args:
            vqg_path: Path to the hdf5 file with questions and images.
            caption_path: Path to the JSON file with image captions.
            tokenizer: Tokenizer for questions and captions.
            transform: Image transformer.
            max_examples: Maximum number of examples to use (for debugging).
            indices: List of indices to use.
        """
        self.vqg_path = vqg_path
        self.caption_path = caption_path
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices

        # Load captions
        with open(caption_path, 'r') as f:
            self.caption_data = json.load(f)
        self.image_to_captions = self._map_image_to_captions()

    def _map_image_to_captions(self):
        """
        Map each image ID to its corresponding list of captions.
        :return: Dictionary mapping image IDs to candidate captions.
        """
        image_to_captions = defaultdict(list)
        for annotation in self.caption_data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']
            image_to_captions[image_id].append(caption)
        return image_to_captions

    def __getitem__(self, index):
        """Returns one data pair (image, question, answer, answer type, captions).
        """
        if not hasattr(self, 'images'):
            self.vqg_data = h5py.File(self.vqg_path, 'r')
            self.questions = self.vqg_data['questions']
            self.answers = self.vqg_data['answers']
            self.answer_types = self.vqg_data['answer_types']
            self.image_indices = self.vqg_data['image_indices']
            self.images = self.vqg_data['images']

        if self.indices is not None:
            index = self.indices[index]

        question = self.questions[index]
        answer = self.answers[index]
        answer_type = self.answer_types[index]
        image_index = self.image_indices[index]
        image = self.images[image_index]

        # Decode question and answer
        question = torch.from_numpy(question)
        question = self.tokenizer.decode(question, skip_special_tokens=True)
        qlength = len(question)

        answer = torch.from_numpy(answer)
        answer = self.tokenizer.decode(answer, skip_special_tokens=True)
        alength = len(answer)

        answer_types_map = {
            0: "activity",
            1: "animal",
            2: "attribute",
            3: "binary",
            4: "color",
            5: "count",
            6: "food",
            7: "location",
            8: "material",
            9: "object",
            10: "other",
            11: "predicate",
            12: "shape",
            13: "spatial",
            14: "stuff",
            15: "time"
        }
        answer_type = answer_types_map.get(answer_type, str(answer_type))

        # Fetch captions for the image
        captions = self.image_to_captions.get(image_index, [])

        #print("Image ID: %s, Caption: %s\n", image_index, captions)

        # Transform the image if a transform is provided
        if self.transform is not None:
            image = self.transform(image)

        return image, question, answer, answer_type, qlength, alength, captions

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        with h5py.File(self.vqg_path, 'r') as vqg_data:
            return vqg_data['questions'].shape[0]


def collate_fn_with_captions(data):
    """Creates mini-batch tensors from the list of tuples.

    Args:
        data: list of tuple (image, question, answer, answer_type, length, captions).

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        questions: List of questions.
        answers: List of answers.
        answer_types: List of answer types.
        qlengths: List of question lengths.
        alengths: List of answer lengths.
        captions: List of captions.
    """
    images, questions, answers, answer_types, qlengths, alengths, captions = zip(*data)
    images = torch.stack(images, 0)

    return images, questions, answers, answer_types, qlengths, alengths, captions


def create_dataset_with_captions(vqg_path, caption_path, config, tokenizer, max_examples=None, indices=None, istrain=True, min_scale=0.5):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if istrain:
        dataset = VQGWithCaptionDataset(vqg_path, caption_path, tokenizer=tokenizer, transform=transform_train,
                                        max_examples=max_examples, indices=indices)
    else:
        dataset = VQGWithCaptionDataset(vqg_path, caption_path, tokenizer=tokenizer, transform=transform_test,
                                        max_examples=max_examples, indices=indices)
    return dataset


def create_sampler(dataset, shuffle, num_tasks, global_rank):
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
    return sampler


def create_loader_with_captions(dataset, sampler, batch_size, num_workers, is_train):
    if is_train:
        shuffle = (sampler is None)
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         sampler=sampler,
                                         drop_last=drop_last,
                                         pin_memory=True,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn_with_captions)
    return loader