import json
from collections import defaultdict

# 이미지 ID에 해당하는 모든 caption 불러오기
class ImageCaptionDataset:
    def __init__(self, json_path):
        """
        Initialize the dataset by loading the JSON file.
        :param json_path: Path to the JSON file containing image-caption data.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_to_captions = self._map_image_to_captions()

    def _map_image_to_captions(self):
        """
        Map each image ID to its corresponding list of captions.
        :return: Dictionary mapping image IDs to candidate captions.
        """
        image_to_captions = defaultdict(list)

        for annotation in self.data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']
            image_to_captions[image_id].append(caption)

        return image_to_captions

    def get_captions_for_image(self, image_id):
        """
        Retrieve all captions for a given image ID.
        :param image_id: Image ID to fetch captions for.
        :return: List of captions corresponding to the image ID.
        """
        return self.image_to_captions.get(image_id, [])