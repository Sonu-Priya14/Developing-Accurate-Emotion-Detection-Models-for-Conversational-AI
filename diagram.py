import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, LSTM, Dense, Dropout, Bidirectional, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Global Constants
EMBEDDING_DIM = 384  # SBERT embedding size
FILTERS = 128
KERNEL_SIZE = 3
POOL_SIZE = 2
HIDDEN_UNITS = 100
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001

# Directory to save diagrams
OUTPUT_DIR = "model_diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_cnn_gru_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same')(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same')(conv_layer)
    gru_layer_1 = GRU(HIDDEN_UNITS, return_sequences=True)(pooling_layer)
    gru_layer_2 = GRU(HIDDEN_UNITS)(gru_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE)(gru_layer_2)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    output_layer = Dense(5, activation='softmax')(dense_layer)  # Adjust num_classes if needed
    return Model(inputs=input_layer, outputs=output_layer)

def build_cnn_bi_gru_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same')(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same')(conv_layer)
    bi_gru_layer_1 = Bidirectional(GRU(HIDDEN_UNITS, return_sequences=True))(pooling_layer)
    bi_gru_layer_2 = Bidirectional(GRU(HIDDEN_UNITS))(bi_gru_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE)(bi_gru_layer_2)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    output_layer = Dense(5, activation='softmax')(dense_layer)
    return Model(inputs=input_layer, outputs=output_layer)

def build_cnn_lstm_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same')(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same')(conv_layer)
    lstm_layer_1 = LSTM(HIDDEN_UNITS, return_sequences=True)(pooling_layer)
    lstm_layer_2 = LSTM(HIDDEN_UNITS)(lstm_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE)(lstm_layer_2)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    output_layer = Dense(5, activation='softmax')(dense_layer)
    return Model(inputs=input_layer, outputs=output_layer)

def build_cnn_bilstm_model():
    input_layer = Input(shape=(EMBEDDING_DIM,), name="Input")
    reshaped_layer = Reshape((1, EMBEDDING_DIM))(input_layer)
    conv_layer = Conv1D(FILTERS, KERNEL_SIZE, activation='relu', padding='same')(reshaped_layer)
    pooling_layer = MaxPooling1D(POOL_SIZE, padding='same')(conv_layer)
    bi_lstm_layer_1 = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(pooling_layer)
    bi_lstm_layer_2 = Bidirectional(LSTM(HIDDEN_UNITS))(bi_lstm_layer_1)
    dropout_layer = Dropout(DROPOUT_RATE)(bi_lstm_layer_2)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    output_layer = Dense(5, activation='softmax')(dense_layer)
    return Model(inputs=input_layer, outputs=output_layer)

# Dictionary of model builders
model_builders = {
    "CNN_GRU": build_cnn_gru_model,
    "CNN_BiGRU": build_cnn_bi_gru_model,
    "CNN_LSTM": build_cnn_lstm_model,
    "CNN_BiLSTM": build_cnn_bilstm_model
}

# Generate model diagrams
for model_name, builder in model_builders.items():
    model = builder()
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}_architecture.png")
    plot_model(model, to_file=model_path, show_shapes=True, show_layer_names=True)
    print(f"Saved {model_name} architecture diagram at {model_path}")
