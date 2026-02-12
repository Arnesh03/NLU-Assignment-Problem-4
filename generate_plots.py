import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Naive Bayes', 'Logistic Regression', 'Linear SVM']
accuracies = [96.10, 95.56, 94.81]

# 1. Model Comparison Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracies, color=['#4285F4', '#EA4335', '#FBBC05'])
plt.ylim(90, 100)
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Machine Learning Models')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval}%', ha='center', va='bottom')
plt.savefig('model_comparison.png')
plt.close()

# 2. Dataset Distribution Pie Chart
dist_labels = ['Sports (1993)', 'Politics (2625)']
counts = [1993, 2625]
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=dist_labels, autopct='%1.1f%%', colors=['#34A853', '#FF6D01'], startangle=140)
plt.title('Dataset Distribution (20 Newsgroups subset)')
plt.savefig('dataset_distribution.png')
plt.close()

# 3. Generating Confusion Matrices for all models
cms = {
    'Naive Bayes': np.array([[545, 4], [32, 343]]),
    'Logistic Regression': np.array([[545, 4], [37, 338]]),
    'Linear SVM': np.array([[533, 16], [34, 341]])
}

filenames = {
    'Naive Bayes': 'cm_nb.png',
    'Logistic Regression': 'cm_lr.png',
    'Linear SVM': 'cm_svm.png'
}

for name, cm in cms.items():
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Politics', 'Sports'])
    plt.yticks(tick_marks, ['Politics', 'Sports'])

    # Annotate
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filenames[name])
    plt.close()

print(f"Plots generated: model_comparison.png, dataset_distribution.png, {', '.join(filenames.values())}")
