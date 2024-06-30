from collections import Counter
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Setting the Seaborn style
sns.set(style="whitegrid")

source_dir = pathlib.Path(r"C:\Users\Milo\OneDrive - Universiteit Utrecht\Period 4\INFOMHCML\Project\BotCL\BotCL\faultyclassifications\Test")

wrong_predictions = []

for dir in source_dir.iterdir():
    if dir.is_dir():
        label, pred = dir.name.split("_")[1].split("as")
        wrong_predictions.append((label, pred))

# Count the occurrences of each wrong prediction
prediction_counter = Counter(wrong_predictions)

# Extract labels and counts for plotting misclassifications
labels, predictions = zip(*prediction_counter.keys())
misclassification_counts = prediction_counter.values()

# Count the total number of wrong classifications per label
label_counter = Counter(label for label, _ in wrong_predictions)

# Sort the labels numerically from 1 to 14 and get the corresponding counts
sorted_labels = sorted(label_counter.keys(), key=lambda x: int(x))
sorted_counts = [label_counter[label] for label in sorted_labels]

# Plotting the total misclassifications per label
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_labels, y=sorted_counts, color="cornflowerblue")
plt.xlabel('True Label', fontsize=14)
plt.ylabel('Count of Misclassifications', fontsize=14)
plt.title('Total Misclassifications Per Label', fontsize=16)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Filter misclassifications with more than 2 wrong predictions
filtered_predictions = [(k, v) for k, v in prediction_counter.items() if v > 2]
filtered_labels_predictions = [f"{label} as {pred}" for label, pred in dict(filtered_predictions).keys()]
filtered_counts = [v for k, v in filtered_predictions]

# Plotting the filtered detailed misclassifications
plt.figure(figsize=(12, 8))
sns.barplot(x=filtered_labels_predictions, y=filtered_counts, color="cornflowerblue")
plt.xlabel('Misclassifications (Label as Prediction)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Most Common Misclassifications (More than 2 wrong predictions)', fontsize=16)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Assuming you have the true labels and predictions in separate lists
true_labels = [label for label, _ in wrong_predictions]
predicted_labels = [pred for _, pred in wrong_predictions]

# Calculate the confusion matrix
labels_sorted = sorted(set(true_labels + predicted_labels), key=lambda x: int(x))
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels_sorted)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=labels_sorted, yticklabels=labels_sorted, cbar_kws={'label': 'Number of Predictions'})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.tight_layout()
plt.show()
