import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class VQAv2Dataset(Dataset):
    def __init__(self, json_file, processor, max_length=512):
        """
        VQAv2 Dataset for BLIP fine-tuning
        
        Args:
            json_file (str): Path to preprocessed JSON file
            processor: BLIP processor for text and image processing
            max_length (int): Maximum sequence length for text
        """
        self.processor = processor
        self.max_length = max_length
        
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {json_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        question = item['question']
        answer = item['answer']
        
        # Process inputs for BLIP VQA
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Process answer as labels for BLIP
        # BLIP expects decoder_input_ids during training
        answer_encoding = self.processor.tokenizer(
            text_target=answer,  # Use text_target for decoder
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50
        )
        
        # Remove batch dimension
        for key in inputs.keys():
            inputs[key] = inputs[key].squeeze(0)
        
        # Add labels for training
        inputs['labels'] = answer_encoding['input_ids'].squeeze(0)
        
        return inputs

def collate_fn(batch):
    """
    Custom collate function for DataLoader
    """
    # Get all keys from the first item
    keys = batch[0].keys()
    
    # Stack all tensors
    collated = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    
    return collated 