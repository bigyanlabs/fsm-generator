# FSM State Diagram Generator — Project Report

## 1. Overview

This project implements a pipeline that converts natural-language system descriptions into formal finite state machine (FSM) specifications and visual diagrams. The implementation uses a GRU-based sequence-to-sequence model for training and inference (model code in model/main.py) and a small curated dataset (model/fsm_dataset.json). The goal is to produce a reproducible, local-capable system that can be extended or replaced by an external LLM in future iterations.

All code and resources mentioned in this report, including the model architecture, training scripts, and dataset, are uploaded to the project's GitHub repository at [Insert your GitHub URL here].

This report documents the motivation and thought process, the design choices, details of the dataset and preprocessing, model architecture, training and inference procedure, sample results and limitations, and steps to reproduce the work. Selected code snippets are included to make the pipeline easy to follow.
---

## 2. Motivation and thought process

Converting informal descriptions into FSMs is useful in testing, specification, control logic design, and education. Two broad approaches were considered:

- Use a pretrained large language model (LLM) via an API, which gives high-quality results quickly but relies on external services and rate limits.
- Train a local seq2seq model on a domain-specific dataset to keep everything self-contained and deterministic.

The project took the second approach first to provide a standalone reference implementation. The guiding priorities were:

- Simple, explainable architecture so components can be replaced later.
- Small, focused dataset with structured FSM outputs to make supervised seq2seq training tractable.
- Clear preprocessing and tokenizer usage so output structure is learnable.
- Exportable tokenizers and saved encoder/decoder models for inference.

Expected trade-offs: limited generalization from a small dataset, but the pipeline is modular and can be replaced with an external LLM for better generality.

---

## 3. Dataset

The dataset is in `model/fsm_dataset.json`. Each item is a small JSON object with `input` and `output` fields. Output strings follow a strict, compact format:

- `states: <comma-separated states> | transitions: <comma-separated transitions>`
- Transitions use either `->` for unconditional transitions or `--event-->` for labeled transitions.

Example entry:
```json
{
  "input": "A vending machine gives output after receiving 10 rupees",
  "output": "states: s0, s5, s10 | transitions: s0 --5rs--> s5, s5 --5rs--> s10, s10 --dispense--> s0"
}
```

The dataset contains common device and control patterns (traffic lights, turnstiles, vending machines, fans, elevators, etc.). It is intentionally compact and structured so a seq2seq model can learn mapping rules and formatting.

---

## 4. Preprocessing

Key preprocessing steps are implemented in `model/main.py` (function `preprocess_text`). Main points:

- Output sequences are wrapped with explicit start and end tokens: `<start> ... <end>`.
- Separate tokenizers are used for input and output (Keras `Tokenizer`), with `oov_token` set.
- Sequences are converted to integer sequences and padded to fixed maximum lengths:
  - `MAX_INPUT_LENGTH = 50`
  - `MAX_OUTPUT_LENGTH = 150`
- Decoder target sequences are prepared by shifting decoder inputs by one timestep (teacher forcing setup).

Relevant snippet:
```python
outputs_with_tokens = ['<start> ' + seq + ' <end>' for seq in outputs]

input_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
output_tokenizer = Tokenizer(filters='', oov_token='<OOV>')

input_tokenizer.fit_on_texts(inputs)
output_tokenizer.fit_on_texts(outputs_with_tokens)

input_sequences = input_tokenizer.texts_to_sequences(inputs)
output_sequences = output_tokenizer.texts_to_sequences(outputs_with_tokens)

encoder_input_data = pad_sequences(input_sequences, maxlen=config.MAX_INPUT_LENGTH, padding='post')
decoder_input_data = pad_sequences(output_sequences, maxlen=config.MAX_OUTPUT_LENGTH, padding='post')

decoder_target_sequences = [seq[1:] + [0] for seq in output_sequences]
decoder_target_data = pad_sequences(decoder_target_sequences, maxlen=config.MAX_OUTPUT_LENGTH, padding='post')
```

Tokenizers are saved to disk so inference can reuse the same vocabulary.

---

## 5. Model architecture

A simple GRU-based sequence-to-sequence architecture is implemented and intended for small datasets:

- Embedding layers for encoder and decoder
- Single-layer GRU encoder producing a final state
- Single-layer GRU decoder that uses the encoder state as its initial state
- Dense softmax output layer over the output vocabulary

Key hyperparameters (in `Config`):
- `LATENT_DIM = 512`
- `EMBEDDING_DIM = 256`
- `BATCH_SIZE = 16`
- `EPOCHS = 200`
- `VALIDATION_SPLIT = 0.15`

Model construction (abbreviated):
```python
encoder_inputs = Input(shape=(None,), name='encoder_input')
encoder_embedding = Embedding(input_vocab_size, config.EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
encoder_gru = GRU(config.LATENT_DIM, return_state=True)
encoder_outputs, encoder_state = encoder_gru(encoder_embedding)

decoder_inputs = Input(shape=(None,), name='decoder_input')
decoder_embedding_layer = Embedding(output_vocab_size, config.EMBEDDING_DIM, mask_zero=True)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_gru = GRU(config.LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)
decoder_outputs = Dense(output_vocab_size, activation='softmax')(decoder_outputs)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
```

Separate encoder and decoder models are constructed for inference (encoder outputs state; decoder accepts previous token and state and returns next-token distribution and new state).

---

## 6. Training

Training uses:
- `optimizer='adam'`
- `loss='sparse_categorical_crossentropy'` with integer targets
- Callbacks: EarlyStopping, ModelCheckpoint (saves best model), ReduceLROnPlateau, TensorBoard

The training function records history and saves it to `saved_models/training_history.pkl`. Models are saved in Keras v3 format (`.keras`) for encoder and decoder.

Training entry point:
```python
history = model.fit(
    [encoder_input, decoder_input],
    np.expand_dims(decoder_target, -1),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    validation_split=config.VALIDATION_SPLIT,
    callbacks=callbacks_list,
    verbose=1
)
```

Notes:
- For small datasets, overfitting is likely; early stopping is important.
- Decreasing batch size and increasing latent/embedding dimension was chosen to give the model capacity while keeping training stable on small data.

---

## 7. Inference

For inference, encoder and decoder are loaded and the decoding loop performs greedy sampling (argmax). The `decode_sequence` function follows standard seq2seq inference:

- Encode input
- Initialize decoder with `<start>` token
- Repeatedly predict next token and advance decoder state
- Stop on `<end>` or length limit

Snippet:
```python
state_value = encoder_model.predict(input_sequence, verbose=0)
target_seq = np.zeros((1, 1)); target_seq[0,0] = output_tokenizer.word_index.get('<start>', 1)
decoded_sentence = ''
while not stop_condition:
    output_tokens, state_value = decoder_model.predict([target_seq, state_value], verbose=0)
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_word = ...  # reverse-lookup from tokenizer
    if sampled_word == '<end>' or len(decoded_sentence.split()) > config.MAX_OUTPUT_LENGTH:
        stop_condition = True
    else:
        decoded_sentence += sampled_word + ' '
    target_seq[0,0] = sampled_token_index
```

The predicted string is then parsed by the UI into states and transitions.

---

## 8. Results and observations

- The model learns the formatting and can reproduce simple FSM patterns and transitions for inputs similar to the training set.
- On out-of-distribution or paraphrased inputs, results vary and may be incorrect or repetitive. This is expected given dataset size.
- The decoder can produce repeated tokens in some runs; adding stronger stopping criteria, temperature sampling, or using beam search can help.
- External LLMs (Gemini, Groq, Hugging Face models) were explored as higher-quality alternatives for production use.

---

## 9. Limitations

- Small dataset limits generalization; expanding dataset or using synthetic data generation is recommended.
- The seq2seq model struggles with long outputs and complicated structural paraphrases.
- No explicit graph grammar or constraint enforcement: model outputs are free text and must be parsed; invalid outputs require cleanup.
- The decoding strategy is greedy; more advanced decoding (beam search, constrained decoding) could improve structure adherence.

---

## 10. Future work

- Replace or augment local seq2seq with an instruction-tuned LLM API for higher fidelity.
- Add constrained decoding or a small FSM grammar validator to ensure outputs always match FSM format.
- Expand dataset with programmatically generated FSM descriptions to cover more patterns.
- Add sequence-level metrics and automated evaluation against ground truth to quantify improvements.
- Improve visual layout (grouping, directed orthogonal edges) and add export to PNG/SVG/GraphML.

---

## 11. Reproduction and usage

1. Setup environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requirements include: tensorflow, numpy, scikit-learn, matplotlib, jupyter (see `requirements.txt` in repo).

2. Train the model
```bash
# from project root
python -c "from model.main import main_train; main_train()"
```

3. Inference
```bash
# single prediction
python -c "from model.main import main_inference; print(main_inference('A light switch toggles between on and off states'))"
```

4. UI
- The repository contains a Flask backend and a static frontend that parses the model output and renders an SVG diagram. Set `GEMINI_API_KEY` if using the external Gemini path. For local model inference, adapt the backend to call `model/main.load_models()` and `model/main.predict_fsm()`.

---

## 12. Selected code references

### load_dataset (model/main.py)
```python
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    inputs = [item['input'] for item in data]
    outputs = [item['output'] for item in data]
    return inputs, outputs
```

### build_seq2seq_model (model/main.py)
(see section 5 above for the compact form)

### decode_sequence (model/main.py)
(see section 7 above for the compact form)

### sample dataset item (model/fsm_dataset.json)
```json
{
  "input": "A fan has three speeds controlled by button",
  "output": "states: off, low, medium, high | transitions: off -> low, low -> medium, medium -> high, high -> off"
}
```

---

## 13. Conclusion

This project provides a complete local pipeline for transforming natural-language descriptions into FSM specifications and visual diagrams. The implementation is small, modular, and intended as a foundation: the model demonstrates the mapping on structured, domain-specific examples, while the UI and parsing components convert model output into clean visual representations. For production-grade quality, the recommended next steps are dataset expansion, constrained decoding, and/or leveraging a modern LLM backend with a constrained output prompt.
.