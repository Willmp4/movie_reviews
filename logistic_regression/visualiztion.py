import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from logistic_regression_sentiment_analysis import main


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['negative', 'positive']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_class_distribution(df):
    class_counts = df['sentiment'].value_counts()

    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

def plot_roc_curve(y_true, y_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs, pos_label='positive')
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    X_test, y_test, y_pred, y_pred_probs, df, model, tfidf_vectorizer = main(train = True)

    plot_confusion_matrix(y_test, y_pred)
    plot_class_distribution(df)
    plot_roc_curve(y_test, y_pred_probs)


