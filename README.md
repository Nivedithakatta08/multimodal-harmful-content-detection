Multimodal Harmful Content Detection



A multimodal deep learning system that classifies memes as offensive or non-offensive by jointly reasoning over image and text content.



Python PyTorch Streamlit License



Table of Contents

Overview

Features

Project Architecture

Dataset

Technologies Used

Project Structure

Installation

Usage

Testing the Dataset

Testing the Model

Training

Inference

Streamlit Application

Model

Results

Limitations

Future Improvements

Application

License

Overview



Memes often carry harmful or offensive meaning through the combination of an image and its overlaid text — neither modality alone always tells the full story. This project tackles that problem with a multimodal deep learning pipeline that fuses:



Visual features extracted with a ResNet18 backbone

Textual features extracted with DistilBERT



to produce a binary classification:



Label	Meaning

Offensive	The meme contains harmful/offensive content

Non-offensive	The meme does not contain harmful/offensive content

Features

Image-based feature extraction using ResNet18

Text-based feature extraction using DistilBERT

Multimodal feature fusion (concatenation + fully connected classifier)

Full training and validation pipeline with checkpointing

Held-out test-set evaluation

Command-line inference on a single image + text pair

Interactive Streamlit web application

Git LFS support for the trained model weights

Project Architecture

text

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



The image and text representations are concatenated and passed through a fully connected classifier to produce the final prediction.



Dataset



This project uses the MultiOFF dataset, which pairs meme images with associated text and offensive/non-offensive labels.



text

MultiOFF\_Dataset/

├── Labelled Images/

└── Split Dataset/

&#x20;   ├── Testing\_meme\_dataset.csv

&#x20;   └── Validation\_meme\_dataset.csv

The validation CSV is further split into training and validation subsets using an 80/20 stratified split.

The testing CSV is reserved exclusively for final model evaluation.

Technologies Used

Category	Tools

Language	Python

Deep Learning	PyTorch, Torchvision

NLP	Hugging Face Transformers, DistilBERT

Vision	ResNet18

Data Handling	Pandas, NumPy, Scikit-learn, Pillow

App / UI	Streamlit

Project Structure

text

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

Installation

Clone the repository

bash

&#x20;  git clone https://github.com/Nivedithakatta08/multimodal-harmful-content-detection.git

Move into the project directory

bash

&#x20;  cd multimodal-harmful-content-detection

Create a virtual environment

bash

&#x20;  python -m venv .venv

Activate the virtual environment Windows (PowerShell):

powershell

&#x20;  .venv\\Scripts\\Activate.ps1



macOS/Linux:



bash

&#x20;  source .venv/bin/activate

Install the required packages

bash

&#x20;  pip install -r requirements.txt

Usage

Testing the Dataset

bash

python test\_dataset.py



Verifies that the dataset loads correctly and that images, tokenized text, and labels are returned in the expected format.



Testing the Model

bash

python test\_model.py



Verifies that the multimodal model accepts image and text inputs and produces classification outputs.



Training

bash

python -m src.train



Training checkpoints are saved to:



text

checkpoints/

Inference



Run inference on a single image and its associated text:



bash

python -m src.inference "path/to/meme.png" "example meme text"



The model returns one of:



text

Prediction: Offensive

text

Prediction: Non-offensive

Streamlit Application



Launch the interactive web app:



bash

streamlit run app.py



The application allows users to:



Upload a meme image

Enter the meme text

Run the multimodal classifier

View the predicted category

View the model's confidence score

Model



The multimodal model consists of three main components:



Image Encoder — ResNet18 extracts visual features from the meme image.

Text Encoder — DistilBERT converts the meme text into a numerical representation.

Classifier — The image and text features are concatenated and passed through fully connected layers to predict the final class.

Results

The initial trained model achieved approximately 60% accuracy on the held-out test set.

The project highlighted the importance of monitoring the gap between training and validation performance: training accuracy increased substantially while validation accuracy remained relatively stable, indicating overfitting.

Limitations

The dataset is relatively small.

Meme meaning can depend heavily on cultural and contextual information.

Text extracted from memes may contain spelling variations, slang, or ambiguous language.

Visual and textual features may not always provide enough context to determine harmful intent.

Model performance may vary on memes that differ significantly from the training distribution.

Future Improvements

Use a larger and more diverse dataset

Apply stronger image augmentation

Fine-tune the pretrained encoders more carefully

Use OCR to automatically extract text from meme images

Experiment with CLIP or other multimodal architectures

Handle class imbalance

Add explainability features (e.g. Grad-CAM, attention visualization)

Improve validation and hyperparameter tuning

Application



This project demonstrates how multimodal machine learning can be applied to content moderation, combining information from multiple modalities instead of relying on image or text classification alone.



License



This project is intended for educational and research purposes.

