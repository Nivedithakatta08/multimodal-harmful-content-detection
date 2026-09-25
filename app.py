import torch
import streamlit as st
from PIL import Image
from transformers import DistilBertTokenizer
from torchvision import transforms

from src.model import MultimodalModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/model_epoch_5.pth"

st.set_page_config(
    page_title="Multimodal Harmful Content Detection",
    page_icon="🛡️"
)


st.title("Multimodal Harmful Content Detection")

st.write(
    "Upload a meme image and enter its text to detect whether "
    "the content is offensive or non-offensive."
)


image_file = st.file_uploader(
    "Upload meme image",
    type=["png", "jpg", "jpeg"]
)


text = st.text_area(
    "Enter meme text"
)


@st.cache_resource
def load_model():

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = MultimodalModel()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer


if image_file is not None:

    image = Image.open(
        image_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded meme",
        use_container_width=True
    )


if st.button("Detect"):

    if image_file is None:

        st.warning(
            "Please upload an image."
        )

    elif not text.strip():

        st.warning(
            "Please enter the meme text."
        )

    else:

        model, tokenizer = load_model()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image_tensor = transform(
            image
        ).unsqueeze(0)

        encoding = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            output = model(
                image_tensor.to(DEVICE),
                encoding["input_ids"].to(DEVICE),
                encoding["attention_mask"].to(DEVICE)
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                prediction
            ].item()


        if prediction == 1:

            st.error(
                "Prediction: Offensive"
            )

        else:

            st.success(
                "Prediction: Non-offensive"
            )


        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )