import torch
from model import Seq2Seq, Encoder, Decoder, Tatoeba_Vocab
import pickle
import spacy

# Función para cargar el vocabulario
def load_vocab(vocab_src_path, vocab_tgt_path):
    with open(vocab_src_path, 'rb') as f:
        vocab_src = pickle.load(f)
    with open(vocab_tgt_path, 'rb') as f:
        vocab_tgt = pickle.load(f)
    return vocab_src, vocab_tgt

# Función para cargar el modelo preentrenado
def load_model(vocab_src,vocab_tgt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = len(vocab_src)  # Asegúrate de cargar los vocabularios correctos
    OUTPUT_DIM = len(vocab_tgt)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    SRC_PAD_IDX = vocab_src['<pad>']
    TGT_PAD_IDX = vocab_tgt['<pad>']

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TGT_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load('prod/tut6-model.pt', map_location=device))
    model.eval()
    return model, device

# Función para traducir una oración
def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len = 50):

    model.eval()
    eng_spacy = spacy.load("en_core_web_sm")

    tokens = [token.text for token in eng_spacy.tokenizer(sentence)] 

    print(tokens)

    src_indexes = [src_vocab[token] for token in tokens]

    print(src_indexes)

    print([src_vocab.idx_to_token[i] for i in src_indexes])

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab['<bos>']]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    trg_tokens = [trg_vocab.idx_to_token[i] for i in trg_indexes]

    return trg_tokens[1:-1], attention