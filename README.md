\# Multimodal Harmful Content Detection

&#x20;

A multimodal deep learning system that classifies memes as \*\*offensive\*\* or \*\*non-offensive\*\* by jointly reasoning over image and text content.

&#x20;

\[!\[Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch\&logoColor=white)](https://pytorch.org/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit\&logoColor=white)](https://streamlit.io/)

\[!\[License](https://img.shields.io/badge/License-Educational%2FResearch-lightgrey.svg)](#license)

&#x20;

\---

&#x20;

\## Table of Contents

&#x20;

\- \[Overview](#overview)

\- \[Features](#features)

\- \[Project Architecture](#project-architecture)

\- \[Dataset](#dataset)

\- \[Technologies Used](#technologies-used)

\- \[Project Structure](#project-structure)

\- \[Installation](#installation)

\- \[Usage](#usage)

&#x20; - \[Testing the Dataset](#testing-the-dataset)

&#x20; - \[Testing the Model](#testing-the-model)

&#x20; - \[Training](#training)

&#x20; - \[Inference](#inference)

&#x20; - \[Streamlit Application](#streamlit-application)

\- \[Model](#model)

\- \[Results](#results)

\- \[Limitations](#limitations)

\- \[Future Improvements](#future-improvements)

\- \[Application](#application)

\- \[License](#license)

\---

&#x20;

\## Overview

&#x20;

Memes often carry harmful or offensive meaning through the \*combination\* of an image and its overlaid text — neither modality alone always tells the full story. This project tackles that problem with a multimodal deep learning pipeline that fuses:

&#x20;

\- \*\*Visual features\*\* extracted with a \*\*ResNet18\*\* backbone

\- \*\*Textual features\*\* extracted with \*\*DistilBERT\*\*

to produce a binary classification:

&#x20;

| Label | Meaning |

|---|---|

| `Offensive` | The meme contains harmful/offensive content |

| `Non-offensive` | The meme does not contain harmful/offensive content |

&#x20;

\## Features

&#x20;

\- Image-based feature extraction using ResNet18

\- Text-based feature extraction using DistilBERT

\- Multimodal feature fusion (concatenation + fully connected classifier)

\- Full training and validation pipeline with checkpointing

\- Held-out test-set evaluation

\- Command-line inference on a single image + text pair

\- Interactive Streamlit web application

\- Git LFS support for the trained model weights

\## Project Architecture

&#x20;

```text

&#x20;  Meme Image                          Meme Text

&#x20;      |                                   |

&#x20;      v                                   v

&#x20;  ResNet18                           DistilBERT

&#x20;      |                                   |

&#x20;      v                                   v

&#x20;Image Features                     Text Features

&#x20;      \\                                   /

&#x20;       \\                                 /

&#x20;        \\-------- Feature Fusion -------/

&#x20;                       |

&#x20;                       v

&#x20;                  Classifier

&#x20;                       |

&#x20;                       v

&#x20;       Prediction: Offensive / Non-offensive

```

&#x20;

The image and text representations are concatenated and passed through a fully connected classifier to produce the final prediction.

&#x20;

\## Dataset

&#x20;

This project uses the \*\*MultiOFF\*\* dataset, which pairs meme images with associated text and offensive/non-offensive labels.

&#x20;

```text

MultiOFF\_Dataset/

├── Labelled Images/

└── Split Dataset/

&#x20;   ├── Testing\_meme\_dataset.csv

&#x20;   └── Validation\_meme\_dataset.csv

```

&#x20;

\- The \*\*validation CSV\*\* is further split into training and validation subsets using an \*\*80/20 stratified split\*\*.

\- The \*\*testing CSV\*\* is reserved exclusively for final model evaluation.

\## Technologies Used

&#x20;

| Category | Tools |

|---|---|

| Language | Python |

| Deep Learning | PyTorch, Torchvision |

| NLP | Hugging Face Transformers, DistilBERT |

| Vision | ResNet18 |

| Data Handling | Pandas, NumPy, Scikit-learn, Pillow |

| App / UI | Streamlit |

&#x20;

\## Project Structure

&#x20;

```text

multimodal-harmful-content-detection/

│

├── MultiOFF\_Dataset/

│   ├── Labelled Images/

│   └── Split Dataset/

│       ├── Testing\_meme\_dataset.csv

│       └── Validation\_meme\_dataset.csv

│

├── src/

│   ├── dataset.py

│   ├── model.py

│   ├── train.py

│   ├── evaluate.py

│   └── inference.py

│

├── checkpoints/

├── models/

│   └── model\_epoch\_5.pth

│

├── app.py

├── test\_dataset.py

├── test\_model.py

├── requirements.txt

├── .gitignore

└── README.md

```

&#x20;

\## Installation

&#x20;

1\. \*\*Clone the repository\*\*

```bash

&#x20;  git clone https://github.com/Nivedithakatta08/multimodal-harmful-content-detection.git

```

&#x20;

2\. \*\*Move into the project directory\*\*

```bash

&#x20;  cd multimodal-harmful-content-detection

```

&#x20;

3\. \*\*Create a virtual environment\*\*

```bash

&#x20;  python -m venv .venv

```

&#x20;

4\. \*\*Activate the virtual environment\*\*

&#x20;  Windows (PowerShell):

```powershell

&#x20;  .venv\\Scripts\\Activate.ps1

```

&#x20;

&#x20;  macOS/Linux:

&#x20;

```bash

&#x20;  source .venv/bin/activate

```

&#x20;

5\. \*\*Install the required packages\*\*

```bash

&#x20;  pip install -r requirements.txt

```

&#x20;

\## Usage

&#x20;

\### Testing the Dataset

&#x20;

```bash

python test\_dataset.py

```

&#x20;

Verifies that the dataset loads correctly and that images, tokenized text, and labels are returned in the expected format.

&#x20;

\### Testing the Model

&#x20;

```bash

python test\_model.py

```

&#x20;

Verifies that the multimodal model accepts image and text inputs and produces classification outputs.

&#x20;

\### Training

&#x20;

```bash

python -m src.train

```

&#x20;

Training checkpoints are saved to:

&#x20;

```text

checkpoints/

```

&#x20;

\### Inference

&#x20;

Run inference on a single image and its associated text:

&#x20;

```bash

python -m src.inference "path/to/meme.png" "example meme text"

```

&#x20;

The model returns one of:

&#x20;

```text

Prediction: Offensive

```

&#x20;

```text

Prediction: Non-offensive

```

&#x20;

\### Streamlit Application

&#x20;

Launch the interactive web app:

&#x20;

```bash

streamlit run app.py

```

&#x20;

The application allows users to:

&#x20;

\- Upload a meme image

\- Enter the meme text

\- Run the multimodal classifier

\- View the predicted category

\- View the model's confidence score

\## Model

&#x20;

The multimodal model consists of three main components:

&#x20;

1\. \*\*Image Encoder\*\* — ResNet18 extracts visual features from the meme image.

2\. \*\*Text Encoder\*\* — DistilBERT converts the meme text into a numerical representation.

3\. \*\*Classifier\*\* — The image and text features are concatenated and passed through fully connected layers to predict the final class.

\## Results

&#x20;

\- The initial trained model achieved approximately \*\*60% accuracy\*\* on the held-out test set.

\- The project highlighted the importance of monitoring the gap between training and validation performance: training accuracy increased substantially while validation accuracy remained relatively stable, indicating \*\*overfitting\*\*.

\## Limitations

&#x20;

\- The dataset is relatively small.

\- Meme meaning can depend heavily on cultural and contextual information.

\- Text extracted from memes may contain spelling variations, slang, or ambiguous language.

\- Visual and textual features may not always provide enough context to determine harmful intent.

\- Model performance may vary on memes that differ significantly from the training distribution.

\## Future Improvements

&#x20;

\- Use a larger and more diverse dataset

\- Apply stronger image augmentation

\- Fine-tune the pretrained encoders more carefully

\- Use OCR to automatically extract text from meme images

\- Experiment with CLIP or other multimodal architectures

\- Handle class imbalance

\- Add explainability features (e.g. Grad-CAM, attention visualization)

\- Improve validation and hyperparameter tuning

\## Application

&#x20;

This project demonstrates how multimodal machine learning can be applied to \*\*content moderation\*\*, combining information from multiple modalities instead of relying on image or text classification alone.

&#x20;

\## License

&#x20;

This project is intended for \*\*educational and research purposes\*\*.

