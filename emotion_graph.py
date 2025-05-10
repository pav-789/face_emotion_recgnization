import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
data = pd.read_csv("emotion_log.csv")

# Count each emotion
emotion_counts = data["Emotion"].value_counts()

# Plot
plt.figure(figsize=(10, 6))
emotion_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Emotion Frequency Chart")
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

# Show graph
plt.show()