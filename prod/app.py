import torch
import streamlit as st
from model import Seq2Seq, Encoder, Decoder, Tatoeba_Vocab
from utils import load_model, translate_sentence, load_vocab
from spacy.cli import download

#Descarga el modelo del tokenizador en ingles
download("en_core_web_sm")

# Cargar los vocabularios
vocab_src, vocab_tgt = load_vocab('prod/vocab_src.pkl', 'prod/vocab_tgt.pkl')

model, device = load_model(vocab_src,vocab_tgt)

# Configuración de Streamlit
st.title('Traductor Inglés-Español')

# Entrada del usuario
input_text = st.text_area("Escribe una frase en inglés para traducir:")

if st.button("Traducir"):
    if input_text:
        # Traducir la frase ingresada
        translation, _ = translate_sentence(input_text, vocab_src, vocab_tgt, model, device)
        st.write("### Traducción:")
        st.write(' '.join(translation))
    else:
        st.write("Por favor, escribe una frase en inglés para traducir.")