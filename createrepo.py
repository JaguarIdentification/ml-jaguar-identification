import os

# Define folder structure
folders = [
    "notebooks/01_data_exploration",
    "notebooks/02_dataset_creation",
    "notebooks/03_modeling",
    "src",
    "models",
    "datasets"
]

# Create directories and add .gitkeep to ensure Git tracks them
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".gitkeep"), "w", encoding="utf-8") as f:
        pass

# Root README content
root_readme = """# Jaguar Identification Project

This project aims to identify individual jaguars using computer vision and deep learning techniques.

It includes:

- A user-facing app to identify jaguars, hosted on Hugging Face Spaces.
- A curated dataset of jaguar images, hosted on Hugging Face.
- An upcoming Kaggle competition to encourage further research and development.

---

## 🚀 App

Try the live demo:  
👉 [Jaguar Identification App on Hugging Face Spaces](https://huggingface.co/spaces/shahabdaiani/jaguar_identification_app)

---

## 📂 Folder Structure

- `notebooks/` – Contains notebooks for data exploration, dataset creation, and model training.
- `src/` – Source code for key functionalities.
- `models/` – Info on model usage.
- `datasets/` – Info and link to the hosted dataset.

---

## 🔗 Hugging Face Resources

- **Dataset**: [jaguaridentification](https://huggingface.co/datasets/jaguaridentification)
- **App**: [Jaguar Identification App](https://huggingface.co/spaces/shahabdaiani/jaguar_identification_app)

---

## 🏆 Kaggle Competition

A Kaggle competition based on this dataset and task will be announced soon. Stay tuned!
"""

# models/README.md content
models_readme = """# Jaguar Identification Model

The trained model used in this project is integrated into the Hugging Face Spaces app:

👉 [Jaguar Identification App](https://huggingface.co/spaces/shahabdaiani/jaguar_identification_app)

The model is not directly published on Hugging Face Model Hub as a standalone entry.
"""

# datasets/README.md content
datasets_readme = """# Jaguar Identification Dataset

The dataset used for training and evaluation is available on Hugging Face Datasets Hub:

👉 [jaguaridentification](https://huggingface.co/datasets/jaguaridentification)

### 📥 How to Load

from datasets import load_dataset dataset = load_dataset("jaguaridentification")


This dataset is used in:

- The [Jaguar Identification App](https://huggingface.co/spaces/shahabdaiani/jaguar_identification_app)
- The upcoming Kaggle competition (announcement coming soon!)
"""

# Write README files with utf-8 encoding
with open("README.md", "w", encoding="utf-8") as f:
    f.write(root_readme)

with open("models/README.md", "w", encoding="utf-8") as f:
    f.write(models_readme)

with open("datasets/README.md", "w", encoding="utf-8") as f:
    f.write(datasets_readme)

print("✅ Project structure, README files, and .gitkeep placeholders created successfully.")
