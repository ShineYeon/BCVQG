"""Loads question answering data and feeds it to the models.
"""

import h5py
import numpy as np

from models.blip import init_tokenizer
from data.randaugment import RandomAugment

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class VQGDataset(Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, tokenizer, transform=None, max_examples=None, indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.dataset = dataset
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.answer_types = annos['answer_types']
            self.image_indices = annos['image_indices']
            self.images = annos['images']
            self.image_paths = annos['image_paths']

        if self.indices is not None:
            index = self.indices[index]
        question = self.questions[index]
        answer = self.answers[index]
        answer_type = self.answer_types[index]
        image_index = self.image_indices[index]
        image = self.images[image_index]
        image_path = self.image_paths[index]

        question = torch.from_numpy(question)
        question = self.tokenizer.decode(question, skip_special_tokens=True)
        qlength = len(question)

        answer = torch.from_numpy(answer)
        answer = self.tokenizer.decode(answer, skip_special_tokens=True)
        alength = len(answer)

        image_path = torch.from_numpy(image_path)
        image_path = self.tokenizer.decode(image_path, skip_special_tokens=True)
        iplength = len(image_path)

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

        # answer_type = self.tokenizer.decode(answer_type, skip_special_tokens=True)

        if self.transform is not None:
            image = self.transform(image)
        return (image_path, image, question, answer, answer_type, qlength, alength)

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
        return annos['questions'].shape[0]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples.

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, question, answer, answer_type, length).
            - image: torch tensor of shape (3, 256, 256).
            - question: torch tensor of shape (?); variable length.
            - answer: torch tensor of shape (?); variable length.
            - answer_type: Int for category label
            - qlength: Int for question length.
            - alength: Int for answer length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        questions: torch tensor of shape (batch_size, padded_length).
        answers: torch tensor of shape (batch_size, padded_length).
        answer_types: torch tensor of shape (batch_size,).
        qindices: torch tensor of shape(batch_size,).
    """
    image_paths, images, questions, answers, answer_types, qlengths, alengths = zip(*data)
    images = torch.stack(images, 0)

    return image_paths, images, questions, answers, answer_types, qlengths, alengths


def create_dataset(dataset, config, tokenizer, max_examples=None, indices=None, istrain=True, min_scale=0.5, demo=False):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.ToPILImage(),
    #     transforms.RandomResizedCrop((config['image_size'], config['image_size']), scale=(0.05, 1.0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.ToPILImage(),
    #     transforms.Resize((config['image_size'], config['image_size'])),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
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

    transform_demo = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        # transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        # transforms.ToTensor(),
        normalize,
    ])

    if istrain:
        vqg = VQGDataset(dataset, tokenizer=tokenizer, transform=transform_train, max_examples=max_examples, indices=indices)
    else:
        vqg = VQGDataset(dataset, tokenizer=tokenizer, transform=transform_test, max_examples=max_examples, indices=indices)
    if demo:
        vqg = VQGDataset(dataset, tokenizer=tokenizer, transform=transform_demo, max_examples=max_examples, indices=indices)
    return vqg


def create_sampler(dataset, shuffle, num_tasks, global_rank):
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
    return sampler


def create_loader(dataset, sampler, batch_size, num_workers, is_train):
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
                                         collate_fn=collate_fn)
    return loader

