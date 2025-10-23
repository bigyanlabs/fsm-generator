"""
FSM State Diagram Generator from Text
GRU-based Seq2Seq Model Implementation
Optimized for Local GPU/CPU Training with <500 samples
"""

import numpy as np
import json
import pickle
import os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

def configure_gpu():
    print("Checking GPU availability...")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"✓ Using GPU: {gpus[0].name}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("✓ No GPU found. Using CPU for training.")
    
    print(f"✓ TensorFlow version: {tf.__version__}")
    print(f"✓ Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    return len(gpus) > 0


class Config:
    """Configuration class for model hyperparameters"""
    LATENT_DIM = 512
    EMBEDDING_DIM = 256
    MAX_INPUT_LENGTH = 50
    MAX_OUTPUT_LENGTH = 150
    BATCH_SIZE = 16
    EPOCHS = 200
    VALIDATION_SPLIT = 0.15
    
    # File paths
    DATASET_PATH = 'fsm_dataset.json'
    MODEL_DIR = 'saved_models'
    MODEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder_model.keras')
    MODEL_DECODER_PATH = os.path.join(MODEL_DIR, 'decoder_model.keras')
    TOKENIZER_INPUT_PATH = os.path.join(MODEL_DIR, 'tokenizer_input.pkl')
    TOKENIZER_OUTPUT_PATH = os.path.join(MODEL_DIR, 'tokenizer_output.pkl')
    TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.pkl')
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
    
    @staticmethod
    def create_directories():
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        print(f"✓ Model directory created/verified: {Config.MODEL_DIR}")

config = Config()


def load_dataset(file_path):
    """
    Load FSM dataset from JSON file
    
    Args:
        file_path: Path to JSON dataset file
        
    Returns:
        inputs: List of input sentences
        outputs: List of output FSM descriptions
    """
    print(f"\nLoading dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    inputs = [item['input'] for item in data]
    outputs = [item['output'] for item in data]
    
    print(f"✓ Loaded {len(inputs)} samples")
    return inputs, outputs


def preprocess_text(inputs, outputs):
    """
    Preprocess input and output text sequences
    
    Args:
        inputs: List of input sentences
        outputs: List of output FSM descriptions
        
    Returns:
        Preprocessed data and tokenizers
    """
    print("\n" + "="*60)
    print("PREPROCESSING TEXT DATA")
    print("="*60)
    
    outputs_with_tokens = ['<start> ' + seq + ' <end>' for seq in outputs]
    
    input_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
    output_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
    
    input_tokenizer.fit_on_texts(inputs)
    output_tokenizer.fit_on_texts(outputs_with_tokens)
    
    input_vocab_size = len(input_tokenizer.word_index) + 1
    output_vocab_size = len(output_tokenizer.word_index) + 1
    
    print(f"✓ Input vocabulary size: {input_vocab_size}")
    print(f"✓ Output vocabulary size: {output_vocab_size}")
    
    input_sequences = input_tokenizer.texts_to_sequences(inputs)
    output_sequences = output_tokenizer.texts_to_sequences(outputs_with_tokens)
    
    encoder_input_data = pad_sequences(
        input_sequences, 
        maxlen=config.MAX_INPUT_LENGTH, 
        padding='post'
    )
    
    decoder_input_data = pad_sequences(
        output_sequences, 
        maxlen=config.MAX_OUTPUT_LENGTH, 
        padding='post'
    )
    
    decoder_target_sequences = []
    for seq in output_sequences:
        decoder_target_sequences.append(seq[1:] + [0])
    
    decoder_target_data = pad_sequences(
        decoder_target_sequences,
        maxlen=config.MAX_OUTPUT_LENGTH,
        padding='post'
    )
    
    print(f"✓ Encoder input shape: {encoder_input_data.shape}")
    print(f"✓ Decoder input shape: {decoder_input_data.shape}")
    print(f"✓ Decoder target shape: {decoder_target_data.shape}")
    
    return (
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        input_tokenizer,
        output_tokenizer,
        input_vocab_size,
        output_vocab_size
    )


def save_tokenizers(input_tokenizer, output_tokenizer):
    """Save tokenizers to disk"""
    with open(config.TOKENIZER_INPUT_PATH, 'wb') as f:
        pickle.dump(input_tokenizer, f)
    
    with open(config.TOKENIZER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(output_tokenizer, f)
    
    print(f"✓ Tokenizers saved successfully")


def load_tokenizers():
    """Load tokenizers from disk"""
    with open(config.TOKENIZER_INPUT_PATH, 'rb') as f:
        input_tokenizer = pickle.load(f)
    
    with open(config.TOKENIZER_OUTPUT_PATH, 'rb') as f:
        output_tokenizer = pickle.load(f)
    
    print("✓ Tokenizers loaded successfully")
    return input_tokenizer, output_tokenizer



def build_seq2seq_model(input_vocab_size, output_vocab_size):
    """
    Build GRU-based Seq2Seq model
    
    Args:
        input_vocab_size: Size of input vocabulary
        output_vocab_size: Size of output vocabulary
        
    Returns:
        training_model: Model for training
        encoder_model: Model for encoding input
        decoder_model: Model for decoding output
    """
    print("\n" + "="*60)
    print("BUILDING SEQ2SEQ MODEL ARCHITECTURE")
    print("="*60)
    
    encoder_inputs = Input(shape=(None,), name='encoder_input')
    
    encoder_embedding = Embedding(
        input_vocab_size,
        config.EMBEDDING_DIM,
        mask_zero=True,
        name='encoder_embedding'
    )(encoder_inputs)
    
    encoder_gru = GRU(
        config.LATENT_DIM,
        return_state=True,
        name='encoder_gru'
    )
    encoder_outputs, encoder_state = encoder_gru(encoder_embedding)
    
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    
    decoder_embedding_layer = Embedding(
        output_vocab_size,
        config.EMBEDDING_DIM,
        mask_zero=True,
        name='decoder_embedding'
    )
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    
    decoder_gru = GRU(
        config.LATENT_DIM,
        return_sequences=True,
        return_state=True,
        name='decoder_gru'
    )
    decoder_outputs, _ = decoder_gru(
        decoder_embedding,
        initial_state=encoder_state
    )
    
    decoder_dense = Dense(
        output_vocab_size,
        activation='softmax',
        name='decoder_output'
    )
    decoder_outputs = decoder_dense(decoder_outputs)
    
   
    training_model = Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs
    )
    
    print("\n✓ Training Model Architecture:")
    training_model.summary()
    
    
    encoder_model = Model(encoder_inputs, encoder_state)
    
    decoder_state_input = Input(shape=(config.LATENT_DIM,))
    decoder_embedding_inf = decoder_embedding_layer(decoder_inputs)
    decoder_outputs_inf, decoder_state_inf = decoder_gru(
        decoder_embedding_inf,
        initial_state=decoder_state_input
    )
    decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
    
    decoder_model = Model(
        [decoder_inputs, decoder_state_input],
        [decoder_outputs_inf, decoder_state_inf]
    )
    
    print("\n✓ Encoder and Decoder models created for inference")
    
    return training_model, encoder_model, decoder_model



def train_model(model, encoder_input, decoder_input, decoder_target):
    """
    Train the Seq2Seq model
    
    Args:
        model: Training model
        encoder_input: Encoder input data
        decoder_input: Decoder input data
        decoder_target: Decoder target data
        
    Returns:
        history: Training history
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_list = []
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    callbacks_list.append(early_stopping)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        config.BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks_list.append(checkpoint)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks_list.append(reduce_lr)
    
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=os.path.join(config.MODEL_DIR, 'logs'),
        histogram_freq=1
    )
    callbacks_list.append(tensorboard)
    
    print(f"✓ Training on {encoder_input.shape[0]} samples")
    print(f"✓ Validation split: {config.VALIDATION_SPLIT * 100}%")
    print(f"✓ Batch size: {config.BATCH_SIZE}")
    print(f"✓ Max epochs: {config.EPOCHS}")
    
    history = model.fit(
        [encoder_input, decoder_input],
        np.expand_dims(decoder_target, -1),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks_list,
        verbose=1
    )
    
    with open(config.TRAINING_HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    
    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"✓ Training history saved to: {config.TRAINING_HISTORY_PATH}")
    
    return history


def plot_training_history(history):
    """Plot training and validation loss/accuracy"""
    print("\n" + "="*60)
    print("PLOTTING TRAINING HISTORY")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(config.MODEL_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training history plots saved as '{plot_path}'")



def decode_sequence(input_seq, encoder_model, decoder_model, 
                      input_tokenizer, output_tokenizer):
    """
    Decode an input sequence to generate FSM description
    
    Args:
        input_seq: Input sequence (text)
        encoder_model: Trained encoder model
        decoder_model: Trained decoder model
        input_tokenizer: Input tokenizer
        output_tokenizer: Output tokenizer
        
    Returns:
        decoded_sentence: Generated FSM description
    """
    input_sequence = input_tokenizer.texts_to_sequences([input_seq])
    input_sequence = pad_sequences(
        input_sequence,
        maxlen=config.MAX_INPUT_LENGTH,
        padding='post'
    )
    
    state_value = encoder_model.predict(input_sequence, verbose=0)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index.get('<start>', 1)
    
    decoded_sentence = ''
    stop_condition = False
    
    while not stop_condition:
        output_tokens, state_value = decoder_model.predict(
            [target_seq, state_value],
            verbose=0
        )
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        sampled_word = None
        for word, index in output_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) > config.MAX_OUTPUT_LENGTH:
            stop_condition = True
        elif sampled_word and sampled_word not in ['<start>', '<OOV>']:
            decoded_sentence += sampled_word + ' '
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
    
    return decoded_sentence.strip()


def predict_fsm(input_text, encoder_model, decoder_model, 
                input_tokenizer, output_tokenizer):
    """
    Generate FSM description from input text
    
    Args:
        input_text: Input scenario description
        encoder_model: Trained encoder model
        decoder_model: Trained decoder model
        input_tokenizer: Input tokenizer
        output_tokenizer: Output tokenizer
        
    Returns:
        fsm_description: Generated FSM description
    """
    print("\n" + "="*60)
    print("GENERATING FSM FROM INPUT")
    print("="*60)
    print(f"Input: {input_text}")
    
    fsm_description = decode_sequence(
        input_text,
        encoder_model,
        decoder_model,
        input_tokenizer,
        output_tokenizer
    )
    
    print(f"Output: {fsm_description}")
    return fsm_description


def batch_predict(test_inputs, encoder_model, decoder_model,
                  input_tokenizer, output_tokenizer, ground_truth=None):
    """
    Predict FSM descriptions for multiple inputs
    
    Args:
        test_inputs: List of input texts
        encoder_model: Trained encoder model
        decoder_model: Trained decoder model
        input_tokenizer: Input tokenizer
        output_tokenizer: Output tokenizer
        ground_truth: Optional ground truth outputs
        
    Returns:
        predictions: List of predicted FSM descriptions
    """
    print("\n" + "="*60)
    print("BATCH PREDICTION")
    print("="*60)
    
    predictions = []
    
    for i, input_text in enumerate(test_inputs):
        prediction = decode_sequence(
            input_text,
            encoder_model,
            decoder_model,
            input_tokenizer,
            output_tokenizer
        )
        predictions.append(prediction)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {input_text}")
        print(f"Predicted: {prediction}")
        if ground_truth:
            print(f"Ground Truth: {ground_truth[i]}")
        print("-" * 60)
    
    return predictions


def save_models(encoder_model, decoder_model):
    """Save encoder and decoder models"""
    encoder_model.save(config.MODEL_ENCODER_PATH, save_format='keras')
    decoder_model.save(config.MODEL_DECODER_PATH, save_format='keras')
    print(f"\n✓ Models saved:")
    print(f"  - Encoder: {config.MODEL_ENCODER_PATH}")
    print(f"  - Decoder: {config.MODEL_DECODER_PATH}")


def load_models():
    """Load encoder and decoder models"""
    if not os.path.exists(config.MODEL_ENCODER_PATH):
        raise FileNotFoundError(f"Encoder model not found: {config.MODEL_ENCODER_PATH}")
    if not os.path.exists(config.MODEL_DECODER_PATH):
        raise FileNotFoundError(f"Decoder model not found: {config.MODEL_DECODER_PATH}")
    
    encoder_model = keras.models.load_model(config.MODEL_ENCODER_PATH, compile=False)
    decoder_model = keras.models.load_model(config.MODEL_DECODER_PATH, compile=False)
    print("✓ Models loaded successfully")
    return encoder_model, decoder_model

# ============================================================================
# SECTION 8: MAIN EXECUTION PIPELINE
# ============================================================================

def main_train():
    """Main training pipeline"""
    print("\n" + "="*80)
    print(" " * 15 + "FSM STATE DIAGRAM GENERATOR")
    print(" " * 20 + "TRAINING PIPELINE")
    print("="*80)
    
    has_gpu = configure_gpu()
    
    config.create_directories()
    
    inputs, outputs = load_dataset(config.DATASET_PATH)
    
    (encoder_input, decoder_input, decoder_target,
     input_tokenizer, output_tokenizer,
     input_vocab_size, output_vocab_size) = preprocess_text(inputs, outputs)
    
    save_tokenizers(input_tokenizer, output_tokenizer)
    
    training_model, encoder_model, decoder_model = build_seq2seq_model(
        input_vocab_size,
        output_vocab_size
    )
    
    history = train_model(
        training_model,
        encoder_input,
        decoder_input,
        decoder_target
    )
    
    plot_training_history(history)
    
    save_models(encoder_model, decoder_model)
    
    print("\n" + "="*80)
    print(" " * 20 + "TRAINING PIPELINE COMPLETED")
    print("="*80)
    
    return encoder_model, decoder_model, input_tokenizer, output_tokenizer


def main_inference(input_text=None):
    """Main inference pipeline"""
    print("\n" + "="*80)
    print(" " * 15 + "FSM STATE DIAGRAM GENERATOR")
    print(" " * 20 + "INFERENCE PIPELINE")
    print("="*80)
    
    try:
        encoder_model, decoder_model = load_models()
        input_tokenizer, output_tokenizer = load_tokenizers()
    except (FileNotFoundError, EOFError) as e:
        print(f"✗ Error during loading: {e}")
        print("Please ensure the model is trained and tokenizers/models are saved.")
        return
    
    if input_text is None:
        test_examples = [
            "A traffic light switches from red to green to yellow in order.",
            "A turnstile unlocks on token and locks again after passing.",
            "A vending machine gives output after receiving 10 rupees.",
        ]
        
        predictions = batch_predict(
            test_examples,
            encoder_model,
            decoder_model,
            input_tokenizer,
            output_tokenizer
        )
    else:
        prediction = predict_fsm(
            input_text,
            encoder_model,
            decoder_model,
            input_tokenizer,
            output_tokenizer
        )
        return prediction
    
    print("\n" + "="*80)
    print(" " * 20 + "INFERENCE PIPELINE COMPLETED")
    print("="*80)



def interactive_mode():
    """Interactive mode for FSM generation"""
    print("\n" + "="*80)
    print(" " * 20 + "INTERACTIVE FSM GENERATOR")
    print("="*80)
    print("\nCommands:")
    print("  - Type any scenario description to generate FSM")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'help' for examples\n")
    
    try:
        encoder_model, decoder_model = load_models()
        input_tokenizer, output_tokenizer = load_tokenizers()
    except (FileNotFoundError, EOFError) as e:
        print(f"✗ Error: {e}")
        print("Please train the model first using main_train()")
        return
    
    while True:
        user_input = input("\n>>> Enter scenario description: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting interactive mode...")
            break
        
        if user_input.lower() == 'help':
            print("\nExample inputs:")
            print("  1. A traffic light switches from red to green to yellow")
            print("  2. A door lock changes state on key insertion")
            print("  3. A vending machine accepts coins until reaching 10 rupees")
            continue
        
        if user_input:
            fsm_output = predict_fsm(
                user_input,
                encoder_model,
                decoder_model,
                input_tokenizer,
                output_tokenizer
            )



if __name__ == "__main__":
    """
    Main entry point for the script
    
    Usage:
        # For training:
        python main.py
        Then run: main_train()
        
        # For inference:
        python main.py
        Then run: main_inference()
        
        # For interactive mode:
        python main.py
        Then run: interactive_mode()
    """
    encoder, decoder, inp_tok, out_tok = main_train()
    print("\n" + "="*80)
    print(" " * 15 + "FSM STATE DIAGRAM GENERATOR")
    print(" " * 25 + "READY TO USE")
    print("="*80)
    