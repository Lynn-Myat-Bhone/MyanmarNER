import streamlit as st
import torch
from models.model import TransformerNER
import pandas as pd
from tools import segmentation
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
# checkpoint = torch.load("model_savefile/TransformerEncoding_model_ver1.pt", map_location=device)
checkpoint_path = hf_hub_download(repo_id="LynnMyatBhone/MyanmarNER", filename="TransformerEncoding_model_ver1.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load mappings
vocab = checkpoint["vocab"]
pos_tag_to_ix = checkpoint["pos_tag_to_ix"]
ner_tag_to_ix = checkpoint["ner_tag_to_ix"]
ix_to_ner_tag = {v: k for k, v in ner_tag_to_ix.items()}

# Dummy embedding matrix
embedding_dim = 300
embedding_matrix = torch.zeros((len(vocab), embedding_dim))

# Load model
model = TransformerNER(
    embedding_matrix=embedding_matrix,
    num_ner_tags=len(ner_tag_to_ix),
    num_pos_tags=len(pos_tag_to_ix)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Predict single sentence
def predict_single_sentence(model, sentence, word2idx, ner_idx2tag, device=device, max_len=128):
    model.eval()
    input_ids = [word2idx.get(word, word2idx["<UNK>"]) for word in sentence]
    length = len(input_ids)
    input_ids += [word2idx["<PAD>"]] * (max_len - length)
    attention_mask = [1]*length + [0]*(max_len - length)

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    mask_tensor = torch.tensor([attention_mask], dtype=torch.bool).to(device)

    with torch.no_grad():
        ner_logits, _ = model(input_tensor, mask=mask_tensor)
        predictions = model.decode(ner_logits, mask=mask_tensor)[0][:length]

    return [(word, ner_idx2tag[tag]) for word, tag in zip(sentence, predictions)]

def format_ner_output(results):
    formatted = []
    for word, tag in results:
        if tag == "O":
            formatted.append(word)
        else:
            # Highlight entities with bold and tag label
            formatted.append(f"**{word}** (_{tag}_)")
    return " ".join(formatted)


# --- Sidebar Navigation ---
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        row-gap: 1.5rem; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("MyNER")
page = st.sidebar.radio("Go to", ["Myanmar NER", "Syllable Tokenization"])

# --- Main Content Based on Nav ---
if page == "Myanmar NER":
    
    st.warning(
        """Disclaimer: This tool is for research purposes. The NER model may not be 100% accurate due to limited data and ongoing development of Myanmar language segmentation.
        Please make sure your input is properly spaced for better named entity recognition."""
    )

    st.markdown("<h4 style='text-align: center;'>Myanmar Named Entity Recognition</h4>", unsafe_allow_html=True)
    user_input = st.text_input("Enter a sentence:", "မောင်မောင် သည် ရန်ကုန် တွင် မွေးဖွားခဲ့သည်")   
    if st.button("Let's Do NER"):
        sentence = user_input.strip().split()
        results = predict_single_sentence(model, sentence, vocab, ix_to_ner_tag)
        st.write("Named Entity Recognition Results:")
        st.markdown(format_ner_output(results))
    
    st.markdown("""
        ---
        ### 📚 Previous Works on Myanmar Language NER

        - **Language Understanding Lab**  
        *Kaung Lwin Thant, Kwankamol Nongpong, Ye Kyaw Thu, Thura Aung, Khaing Hsu Wai, Thazin Myint Oo*
        *myNER: Contextualized Burmese Named Entity Recognition with Bidirectional LSTM and fastText Embeddings via Joint Training with POS Tagging*,
        the International Conference on Cybernetics and Innovations (ICCI 2025), April 2-4, Pattaya Chonburi, Thailand pp.
        [🔗 github link] (https://github.com/ye-kyaw-thu/myNER.git)

        - **Hsu Myat Mo et al.**  
        *CRF-Based Named Entity Recognition for Myanmar Language,*  
        in *Genetic and Evolutionary Computing (ICGEC 2016)*, J. S. Pan et al., Eds.,  
        Advances in Intelligent Systems and Computing, vol. 536. Cham: Springer, 2017.  
        [🔗 Springer Link](https://link.springer.com/chapter/10.1007/978-3-319-48490-7_24)

        - **Hsu Myat Mo and Khin Mar Soe**  
        *Named Entity Recognition for Myanmar Language,*  
        in *Proceedings of the 2022 International Conference on Communication and Computer Research (ICCR 2022)*,  
        Sookmyung Women’s University, Seoul, Korea, 2022.  
        [🔗 ResearchGate Link](https://www.researchgate.net/publication/379828999_Named_Entity_Recognition_for_Myanmar_Language)

        - **Hsu Myat Mo and Khin Mar Soe**  
        *Syllable-based Neural Named Entity Recognition for Myanmar Language.*  
        Last modified 2019.  
        [🔗 arXiv:1903.04739](https://arxiv.org/abs/1903.04739)
        """)
    st.markdown("""
        ---
        ### Developed by
        - LynnMyat Bhone: (https://github.com/Lynn-Myat-Bhone)
        - Thuta Nyan : (https://github.com/ThutaNyan788)
        - Shin Thant Phyo : (https://github.com/NanGyeThote)

        """
    )

        
    
elif page == "Syllable Tokenization":
    st.markdown("<h4 style='text-align: center;'>Myanmar Syllable Tokenization</h4>", unsafe_allow_html=True)
    user_input = st.text_input("Enter a sentence:", "မောင်မောင် သည် ရန်ကုန် တွင် မွေးဖွားခဲ့သည်")
    
    if st.button('Submit'):
        seg_char = segmentation.segment_characters(user_input)
        res  = " ".join(seg_char)
        st.write("Tokenized Syllables:", res)
        
    st.markdown("""
        ---
        ### Developed by
        - Thihan Soe: (https://github.com/Yoinami)

        """
    )
