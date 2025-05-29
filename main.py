import streamlit as st
import torch
from models.model import TransformerNER
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load checkpoint
checkpoint = torch.load("model_savefile/TransformerEncoding_model_ver1.pt", map_location=device)

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

# --- Streamlit UI ---
select = pd.DataFrame()
select['topics'] = ['Myanmar NER', 'syllable-tokenization']
option = st.sidebar.selectbox(
    '',select['topics'])

if(option == "Myanmar NER"):
    st.markdown("<h4 style='text-align: center;'>Myanmar Named Entity Recognition</h4>", unsafe_allow_html=True)
    st.write("\n")
    user_input = st.text_input("Sentence", "မောင်မောင် သည် ရန်ကုန် တွင် မွေးဖွားခဲ့သည်")
    if st.button("Let's Do NER"):
        sentence = user_input.strip().split()
        results = predict_single_sentence(model, sentence, vocab, ix_to_ner_tag)

        st.subheader("Named Entity Recognition Results:")
        st.markdown(format_ner_output(results))

    

