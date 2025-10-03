import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import base64

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Ingredient Classifier",
    page_icon="🍴",
    layout="centered"
)

# -------------------------------
# Set background image
# -------------------------------
def set_background(png_file):
    with open(png_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(0,0,0,0);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_background("Static/background.png")  # Replace with your background image path

# -------------------------------
# Load model and tokenizer
# -------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = BertForSequenceClassification.from_pretrained("bert_model")
tokenizer = BertTokenizer.from_pretrained("bert_model")
model = model.to(device)
model.eval()

# -------------------------------
# Page Header
# -------------------------------
st.markdown("<h1 style='text-align: center; color: black;'>Ingredient Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Predict if allergen is present in your recipe</h3>", unsafe_allow_html=True)

# Header image
st.image("Static/image.jpg", use_container_width=True)  # Replace with your header image

# -------------------------------
# Input Section
# -------------------------------
st.markdown("### Enter your ingredients below:")
ingredient_input = st.text_area(
    "Ingredients (separate by commas):",
    placeholder="E.g., Sugar, Milk, Cocoa, Butter",
    height=100
)

# Optional illustration image
st.image("Static/image.jpg", width=300)  # Replace with your illustration image

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):
    if ingredient_input.strip() == "":
        st.warning("Please enter some ingredients to predict!")
    else:
        # Tokenize input
        encoding = tokenizer(
            [ingredient_input],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Map label for allergens
        label_map = {0: "Allergen Not Present", 1: "Allergen Present"}
        st.success(f"Predicted label: **{label_map[predicted_class]}**")
        st.info(f"Confidence → Class 0: {probs[0]:.2f}, Class 1: {probs[1]:.2f}")

        # Optional result image
        st.image("Static/image.jpg", width=300)  # Replace with your result image

# -------------------------------
# Footer Section
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Powered by BERT & Streamlit</p>", unsafe_allow_html=True)
