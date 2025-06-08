import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed data
train_df = pd.read_csv('Train_Preprocess.csv')
test_df = pd.read_csv('Test_Preprocess.csv')

# 2. Create synthetic time-series sequences
def create_sequences(data, seq_length=10):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length, :-1].values
        label = data.iloc[i+seq_length, -1]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

SEQ_LENGTH = 10
FEATURES = train_df.shape[1] - 1

X_train, y_train = create_sequences(train_df, SEQ_LENGTH)
X_test, y_test = create_sequences(test_df, SEQ_LENGTH)

# 3. Train-Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# 4. Build GRU model
model = Sequential([
    GRU(128, input_shape=(SEQ_LENGTH, FEATURES), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    
    GRU(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

# 5. Compile the model
optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])

# 6. Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
    ModelCheckpoint('best_gru_model.h5', monitor='val_auc', save_best_only=True, mode='max')
]

# 7. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    class_weight={0: 1, 1: 2.5}  # Higher weight for positive class
)

# 8. Evaluation function
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model on test set...")
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Test AUC: {results[4]:.4f}")
    
    # Predictions and classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Backorder', 'Backorder'],
                yticklabels=['No Backorder', 'Backorder'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('GRU Model - Confusion Matrix')
    plt.show()

evaluate_model(model, X_test, y_test)

# 9. Plot training history
def plot_history(history):
    plt.figure(figsize=(14,5))
    
    # Plot accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1,2,2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# 10. Save final model
model.save('backorder_gru_model.h5')
print("GRU model saved successfully!")
