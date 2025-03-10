# Plot confusion_matrix for individual label
from sklearn.metrics import multilabel_confusion_matrix
mlcm = multilabel_confusion_matrix(y_test, y_predict)

labels = [0, 1, 2, 3, 4, 5]


fig, axes = plt.subplots(2, 3, figsize=(15, 8))  
axes = axes.flatten()  

for i, (ax, cm, label) in enumerate(zip(axes, mlcm, labels)):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], 
                yticklabels=["Neg", "Pos"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix for Label {label}")

plt.tight_layout()  
plt.show()

# Calculate Precision/Recall for individual label
import numpy as np
from sklearn.metrics import precision_score, recall_score


labels = [0, 1, 2, 3, 4, 5]
precision_per_label = precision_score(y_test, y_predict, average=None)  
recall_per_label = recall_score(y_test, y_predict, average=None)        


for label, prec, rec in zip(labels, precision_per_label, recall_per_label):
    print(f"Label {label}: Precision = {prec:.3f}, Recall = {rec:.3f}")