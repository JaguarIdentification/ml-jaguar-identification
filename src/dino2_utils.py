#pip install pytorch-metric-learning
try:
    from pytorch_metric_learning import losses
    _has_metric_learning = True
except ImportError:
    _has_metric_learning = False
    import warnings
    warnings.warn("pytorch_metric_learning not installed. Some functions may not work.")
    
import os
import numpy as np
from PIL import Image
from pathlib import Path
import random
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import fiftyone as fo


# Define the custom DINOv2 model 
class DINOv2ArcFace(nn.Module):
    """
    DINOv2 model with optional ArcFace loss.
    ArcFace components are initialized only if `usage='finetune'`.
    By setting `usage='embeddings'`, you can use the model to extract embeddings.
    By setting `usage='classifier'`, you can classify without ArcFace loss.  
    """
    def __init__(self, usage='finetune', num_classes=32, embedding_dim=512, margin=0.5, scale=64.0):
        super().__init__()
        self.usage = usage
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.dropout = nn.Dropout(p=0.5) 

        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        if self.usage == 'finetune':
            # Embedding layer and ArcFace loss
            self.embedding = nn.Linear(self.backbone.config.hidden_size, self.embedding_dim)
            self.arcface = losses.ArcFaceLoss(
                num_classes=self.num_classes,
                embedding_size=self.embedding_dim,
                margin=self.margin,
                scale=self.scale
            )

        elif self.usage == 'classifier':
            # Freeze the backbone and add a classifier layer
            self.backbone.requires_grad_(False)
            in_features = self.backbone.config.hidden_size
            self.classifier = nn.Linear(in_features, self.num_classes)

        elif self.usage == 'embeddings':
            # Only use embedding layer for extracting features
            self.embedding = nn.Linear(self.backbone.config.hidden_size, self.embedding_dim)

    def forward(self, x, labels=None):
        # Extract features using CLS token (first token)
        features = self.backbone(x).last_hidden_state[:, 0, :]  # CLS token

        if self.usage == 'finetune' and labels is not None:
            embeddings = F.normalize(self.embedding(features), p=2, dim=1)
            # If in finetune mode, calculate ArcFace loss
            loss = self.arcface(embeddings, labels)
            return embeddings, loss

        elif self.usage == 'classifier':
            # If in classifier mode, return logits (no embedding needed)
            features = self.dropout(features)
            logits = self.classifier(features)
            return logits

        else:
            # Otherwise, return the embeddings (for 'embeddings' usage mode)
            embeddings = F.normalize(self.embedding(features), p=2, dim=1)
            return embeddings


# Custom Datasets
class JaguarDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images and labels.
    This dataset can use either a Hugging Face processor or a PyTorch transform for preprocessing.
    """
    def __init__(self, image_paths, labels, processor=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor  # Hugging Face processor
        self.transform = transform  # Custom PyTorch transform

        # Ensure only one preprocessing method is provided
        assert not (processor is not None and transform is not None), \
            "Use either 'processor' (Hugging Face) or 'transform' (PyTorch), not both."

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = torch.tensor(self.labels[idx])

        if self.processor is not None:
            # Use Hugging Face processor
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), label  

        elif self.transform is not None:
            # Use custom transform (e.g., padding + resize)
            return self.transform(image), label
        else:
            raise ValueError("No preprocessing method provided!")
        

# Define A Data Processor to Store Train/Test Splits
class JaguarDatasetProcessor:
    def __init__(self, add_dataset_unknown_jaguars=True, add_unknown_jaguars=False, unknown_jaguars_dir=None, train_ratio=0.7):
        """
        Initialize the JaguarDatasetProcessor class.
        
        Args:
        - add_dataset_unknown_jaguars: Whether to include unknown jaguars from the dataset (default True)
        - add_unknown_jaguars: Whether to include external unknown jaguars (default False)
        - unknown_jaguars_dir: Path to directory containing external unknown jaguar images
        - train_ratio: Proportion of unknown jaguars for training (default 0.7)
        """
        self.add_dataset_unknown_jaguars = add_dataset_unknown_jaguars
        self.add_unknown_jaguars = add_unknown_jaguars
        self.train_ratio = train_ratio
        
        if self.add_unknown_jaguars:
            if unknown_jaguars_dir is None:
                raise ValueError("unknown_jaguars_dir must be provided if add_unknown_jaguars is True.")
            self.unknown_jaguars_dir = Path(unknown_jaguars_dir)
        else:
            self.unknown_jaguars_dir = None
        
        # Initialize all storage lists
        self.train_image_paths = []
        self.train_labels_str = []
        self.test_image_paths = []
        self.test_labels_str = []
        self.name_to_int = {}  
        self.num_jaguars = 0

    def process_dataset_jaguars(self, dataset):
        """Process known and unknown jaguars from the dataset for train and test sets (unknown jaguars are only added to the train set)."""
        # Process known jaguars - train set
        train_dataset = dataset.match({"testtrainsplit_cosine_similarity": "train"})
        self.train_image_paths.extend(train_dataset.values("filepath"))
        self.train_labels_str.extend([l.label for l in train_dataset.values("ground_truth")])
        
        # Process known jaguars - test set
        test_dataset = dataset.match({"testtrainsplit_cosine_similarity": "test"})
        self.test_image_paths.extend(test_dataset.values("filepath"))
        self.test_labels_str.extend([l.label for l in test_dataset.values("ground_truth")])
        
        # Process dataset unknown jaguars if enabled
        if self.add_dataset_unknown_jaguars: 
            train_dataset_unknown = dataset.match({"testtrainsplit_cosine_similarity": "unknown"})
            self.train_image_paths.extend(train_dataset_unknown.values("filepath"))
            self.train_labels_str.extend(["unknown"] * len(train_dataset_unknown)) 

    def process_unknown_jaguars(self):
        """Process unknown jaguars (train/test split and integration into known jaguars)."""
        if not self.add_unknown_jaguars:
            return
        
        if self.unknown_jaguars_dir is None:
            raise ValueError("unknown_jaguars_dir must be provided if add_unknown_jaguars is True.")
        
        all_unknown_jaguars = list(self.unknown_jaguars_dir.glob("*.jpg"))

        # Shuffle and split into train/test (70/30)
        random.shuffle(all_unknown_jaguars)
        num_train = int(len(all_unknown_jaguars) * self.train_ratio)

        unknown_jaguars_train = all_unknown_jaguars[:num_train]
        unknown_jaguars_test = all_unknown_jaguars[num_train:]

        # Add unknown jaguars to train set
        self.train_image_paths.extend([str(path) for path in unknown_jaguars_train])
        self.train_labels_str.extend(["Unknown"] * len(unknown_jaguars_train))

        # Add unknown jaguars to test set
        self.test_image_paths.extend([str(path) for path in unknown_jaguars_test])
        self.test_labels_str.extend(["Unknown"] * len(unknown_jaguars_test))

    def create_label_mapping(self):
        """Create label mapping and convert string labels to integers."""
        # Combine all labels to get complete set of names
        all_labels = self.train_labels_str + self.test_labels_str
        jaguar_names = set(all_labels)
        sorted_jaguar_names = sorted(jaguar_names)
        self.num_jaguars = len(sorted_jaguar_names)

        # Create a dictionary mapping each name to a unique integer
        self.name_to_int = {name: idx for idx, name in enumerate(sorted_jaguar_names)}

        # Convert string labels to integers
        self.train_labels = [self.name_to_int[name] for name in self.train_labels_str]
        self.test_labels = [self.name_to_int[name] for name in self.test_labels_str]

    def verify_data(self):
        """Print dataset and label information."""
        print(f"Total jaguar identities: {self.num_jaguars}")
        print(f"Train set: {len(self.train_image_paths)} images")
        print(f"Test set: {len(self.test_image_paths)} images")
        print(f"Label mapping: {self.name_to_int}")

    def process(self, dataset):
        """Full pipeline to process the dataset."""
        self.process_dataset_jaguars(dataset)
        self.process_unknown_jaguars()  # Will do nothing if add_unknown_jaguars is False
        self.create_label_mapping()
        self.verify_data()


# Define custom transform to pad images to square
def pad_to_square(image, fill=0):
    """
    Pad the image to a square by adding a border of the specified color.
    The color is specified by the fill parameter.
    """
    w, h = image.size
    max_dim = max(w, h)
    padded = Image.new(image.mode, (max_dim, max_dim), fill)
    padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    return padded


# Function to create the transformation pipeline
def setup_transform(use_padding=True, use_augmentation=False):
    """
    Create a transformation pipeline for the images.
    
    Args:
        use_padding (bool): Whether to pad images to square aspect ratio
        use_augmentation (bool): Whether to apply data augmentations
        
    Returns:
        torchvision.transforms.Compose: The transformation pipeline
    """
    # Base transforms (applied to all images)
    base_transforms = []
    
    # Padding/non-padding logic
    if use_padding:
        base_transforms.append(lambda img: pad_to_square(img))  # Gray padding
        base_transforms.append(transforms.Resize(224))
    else:
        base_transforms.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    
    # Augmentation transforms (only applied if use_augmentation=True)
    augmentation_transforms = []
    if use_augmentation:
        augmentation_transforms.extend([
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ])
    
    # Final transforms (applied to all images)
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # Combine all transforms
    transform = transforms.Compose(
        base_transforms + 
        augmentation_transforms + 
        final_transforms
    )
    
    return transform


# Function to get the embedding of an image
def get_embedding(image_path, model, transform):
    """
    Get the embedding of an image using the DINOv2 model.
    """
    try:
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)

        # Move to same device as model
        device = next(model.parameters()).device
        img_unsqueezed = img_t.unsqueeze(0).to(device)

        # Forward pass + L2-normalize
        with torch.no_grad():
            embedding = model(img_unsqueezed)

        return embedding.squeeze().cpu().numpy()  # Return as numpy array

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None  # Skip failed images


# Define Custom Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)  # Log softmax for the inputs
        ce_loss = self.ce(inputs, targets)  # Compute the standard CrossEntropyLoss
        p = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = (1 - p) ** self.gamma * ce_loss  # Focal Loss formula
        return focal_loss.mean()  # Return the mean of the focal loss