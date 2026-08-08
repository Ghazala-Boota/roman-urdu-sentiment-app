
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Roman Urdu Sentiment Analysis", page_icon="🇵🇰", layout="wide")

@st.cache_resource
def load_model():
    model_name = "GhazalaBoota/roman-urdu-sentiment-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
labels = ["Negative", "Neutral", "Positive"]

st.title("🇵🇰 Roman Urdu Sentiment Analysis")
st.markdown("### Fine-tuned Multilingual BERT for Low-Resource NLP")

tab1, tab2, tab3 = st.tabs(["Try it", "Model Performance", "About"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        text = st.text_area("Roman Urdu Text", placeholder="Jaise: ye cheez bohat acha hai", height=120)
        predict = st.button("Predict Sentiment", type="primary")
    
    with col2:
        if predict and text.strip():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).numpy()[0]
            
            pred_label = labels[probs.argmax()]
            st.success(f"Predicted Sentiment: **{pred_label}**")
            
            for label, score in zip(labels, probs):
                st.write(f"{label}")
                st.progress(float(score))
                st.caption(f"{score:.1%}")
        elif predict:
            st.warning("Please koi text likhein.")

with tab2:
    st.markdown("""
    ## Comparison: Baseline vs Fine-tuned BERT
    
    | Model | Accuracy | Negative F1 | Neutral F1 | Positive F1 |
    |-------|----------|--------------|------------|--------------|
    | TF-IDF + Logistic Regression | 64% | 0.56 | 0.69 | 0.63 |
    | **Fine-tuned BERT (ours)** | **65%** | **0.60** | 0.69 | **0.65** |
    
    **Key finding**: Fine-tuned BERT shows the largest improvement on 
    Negative sentiment detection, the class the baseline struggled 
    with most.
    """)

with tab3:
    st.markdown("""
    ## Methodology
    
    **Dataset**: Roman Urdu Data Set (Sharf, 2017, UCI ML Repository)  
    19,678 cleaned records after removing duplicates and fixing label errors.
    
    **Pipeline**:
    1. Data cleaning, removed 543 duplicates, fixed typo labels, handled missing values
    2. Baseline, TF-IDF (5000 features, bigrams) + Logistic Regression  
    3. Fine-tuning, bert-base-multilingual-cased, 6 epochs, learning rate 2e-5
    4. Best checkpoint selected by validation F1, avoided overfitting seen in later epochs
    
    **Model card**: [huggingface.co/GhazalaBoota/roman-urdu-sentiment-bert](https://huggingface.co/GhazalaBoota/roman-urdu-sentiment-bert)
    """)
