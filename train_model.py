import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading recorded gesture data...")

# 1. Load your spreadsheet
try:
    df = pd.read_csv('two_hand_data.csv')
except FileNotFoundError:
    print("Error: Could not find two_hand_data.csv! Make sure the filename matches.")
    exit()

# 2. Separate the features (coordinates) from the target labels (jutsu names)
X = df.drop('Label', axis=1) # Everything except the 'Label' column
y = df['Label']              # Just the 'Label' column

# 3. Split data into a Training set and a Testing set
# We train the AI on 80% of your data and hold back 20% to test it like a final exam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data parsed! Training the Neural Network on {len(X_train)} samples...")

# 4. Define the Multi-Layer Perceptron (Neural Network)
# hidden_layer_sizes=(64, 32) creates two hidden layers of processing nodes
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

# 5. The Training Process (The AI searches for mathematical patterns here)
model.fit(X_train, y_train)

# 6. Evaluate the model's intelligence
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n--- TRAINING SUMMARY ---")
print(f"Model Classification Accuracy: {accuracy * 100:.2f}%")

# 7. Save the trained brain to a file
with open('jutsu_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Trained model brain successfully saved as 'jutsu_model.pkl'!")