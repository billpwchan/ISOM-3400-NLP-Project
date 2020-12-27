import itertools
import tkinter as Tkinter
from tkinter import *
from tkinter.ttk import *

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn import metrics, preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import *


def dataPreparation():
    global vect
    vect = CountVectorizer(max_features=1500, max_df=0.9,
                           stop_words=stopwords.words('english'))

    review['bin_stars'] = review['stars'].map({5: 1, 1: 0})
    global X_train, X_test, y_train, y_test
    if varBinMul.get() == 1:
        X_train, X_test, y_train, y_test = train_test_split(
            review['text'], review['stars'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(review[(review.stars == 1) | (
            review.stars == 5)]['text'], review[(review.stars == 1) | (review.stars == 5)]['bin_stars'])
    global X_train_cuf, X_test_cuf, y_train_cuf, y_test_cuf
    X_train_cuf, X_test_cuf, y_train_cuf, y_test_cuf = train_test_split(
        review[['cool', 'useful', 'funny']], review['stars'])

    global X_train_t
    X_train_t = vect.fit_transform(X_train)
    global X_test_t
    X_test_t = vect.transform(X_test)
    # Remove Common&Freq words but irrelevant to the meaning words. (Term Frequency-InversDocument Frequency)
    tfidf_transform = TfidfTransformer()
    global X_train_tfidf
    X_train_tfidf = tfidf_transform.fit_transform(X_train_t)
    global X_test_tfidf
    X_test_tfidf = tfidf_transform.fit_transform(X_test_t)

# 10-fold Cross Validation Evaluation


def modelAnalysis(vectorizer, X_train, Y_train):
    X_train_t = vectorizer.fit_transform(X_train)
    models = [
        RandomForestClassifier(),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(),
        SGDClassifier(loss='log', max_iter=100000, tol=1e-3),
        MLPClassifier(solver='lbfgs', alpha=1e-5),
        AdaBoostClassifier()
    ]
    CV = 10
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(
            model, X_train_t, Y_train, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(
        entries, columns=['model_name', 'fold_idx', 'accuracy'])
    sns.set_style("dark")
    sns.boxplot(x='model_name', y='accuracy', data=cv_df, palette="deep")
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def predictionReport(y_pred, y_test, labels, y_pred_proba=[]):
    report_data = []
    lines = metrics.classification_report(y_test, y_pred).split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    print(metrics.classification_report(y_test, y_pred))
    print("Predication Accuracy: %.10f" %
          metrics.accuracy_score(y_test, y_pred))
    # messagebox.showinfo("Prediction Accuracy", metrics.accuracy_score(y_test, y_pred))
    plt.figure(1)
    plot_confusion_matrix(metrics.confusion_matrix(y_test, y_pred), classes=labels,
                          title='Confusion Matrix with Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred)))
    plt.tight_layout()
    if (len(y_pred_proba) != 0 and varBinMul.get() == 0):
        plt.figure(2)
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %s)' % str(auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
    plt.show()


def gridSearchModule(trainData, targetData, model, param_grid):
    grid = GridSearchCV(model, param_grid, cv=10,
                        scoring='accuracy', n_jobs=-1)
    grid.fit(trainData, targetData)
    return grid.grid_scores_, grid.best_estimator_


def KNN(trainData, targetData, testData, actualResult, classes):
    param_grid = dict(n_neighbors=list(range(1, 2)))
    scores, bestEstimator = gridSearchModule(
        trainData, targetData, KNeighborsClassifier(n_neighbors=100), param_grid)
    print(scores, bestEstimator)

    knn = bestEstimator
    knn.fit(trainData, targetData)
    y_pred_class = knn.predict(testData)
    predictionReport(y_pred_class, actualResult, classes)


def LR(trainData, targetData, testData, actualResult, classes):
    param_grid = [
        {'penalty': ['l1'], 'solver': [
            'liblinear', 'saga'], 'max_iter': [50000]},
        {'penalty': ['l2'], 'solver': ['newton-cg',
                                       'lbfgs', 'sag'], 'max_iter': [50000]},
    ]
    scores, bestEstimator = gridSearchModule(
        trainData, targetData, LogisticRegression(), param_grid)
    print(scores, bestEstimator)

    lr = bestEstimator
    lr.fit(trainData, targetData)
    y_pred_class = lr.predict(testData)
    y_pred_proba = lr.predict_proba(testData)[::, 1]
    predictionReport(y_pred_class, actualResult, classes, y_pred_proba)


def MNB(trainData, targetData, testData, actualResult, classes):
    param_grid = [{'alpha': [0.0000, 0.0001, 0.001, 0.01, 0.1, 1]}]
    scores, bestEstimator = gridSearchModule(
        trainData, targetData, MultinomialNB(), param_grid)
    print(scores, bestEstimator)

    nb = bestEstimator
    nb.fit(X_train_t, y_train)
    y_pred_class = nb.predict(X_test_t)
    predictionReport(y_test, y_pred_class, classes)


def SGDLog(trainData, targetData, testData, actualResult, classes):
    param_grid = [
        {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'loss': [
            'log'], 'max_iter': [50000]}
    ]
    scores, bestEstimator = gridSearchModule(
        trainData, targetData, SGDClassifier(), param_grid)
    print(scores, bestEstimator)

    svm = bestEstimator

    svm.fit(trainData, targetData)
    y_pred_class = svm.predict(testData)

    predictionReport(y_test, y_pred_class, classes)


def SVCLinear(trainData, targetData, testData, actualResult, classes):
    param_grid = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'max_iter': [50000], 'tol': [0.0001, 0.001, 0.01, 0.1]}]
    scores, bestEstimator = gridSearchModule(
        trainData, targetData, SVC(), param_grid)
    print(scores, bestEstimator)

    svc = bestEstimator

    svc.fit(trainData, targetData)
    y_pred_class = svc.predict(testData)
    predictionReport(y_test, y_pred_class, classes)


def main():
    root = Tkinter.Tk()
    nltk.download('stopwords')
    global review
    review = pd.read_csv("review.csv")
    global varBinMul
    varBinMul = IntVar()
    varBinMul.set(0)
    dataPreparation()

    root.title("Text-Mining Model Training & Evaluating Utility")
    root.configure(bg='#ffffff')

    Tkinter.ttk.Label(
        root, text="Model Training (With Parameter Tuning)").grid(row=0, column=0, columnspan=2, sticky=N+S+E+W)

    c_BinMul = Checkbutton(root, text="Binary/Multi Classifier (Tick to enable Multi-class)",
                           variable=varBinMul, command=dataPreparation).grid(row=1, column=1, sticky=W)
    btnKNNCUF = Tkinter.ttk.Button(root, text="KNN (With using Cool, Useful & Funny)", command=lambda: KNN(
        X_train_cuf, y_train_cuf, X_test_cuf, y_test_cuf, ['1', '2', '3', '4', '5']), style='C.TButton').grid(row=1, column=0, sticky=N+S+E+W)

    varTFIDF_LR = IntVar()
    c_LR = Checkbutton(root, text="Logistic Regression(TF-IDF)",
                       variable=varTFIDF_LR).grid(row=2, column=1, sticky=W)
    btnLR = Tkinter.ttk.Button(root, text="Logistic Regression", command=lambda: LR(
        X_train_t if varTFIDF_LR.get() == 0 else X_train_tfidf, y_train, X_test_t if varTFIDF_LR.get() == 0 else X_test_tfidf, y_test, ['1', '2'] if varBinMul.get() == 0 else ['1', '2', '3', '4', '5']), style='C.TButton').grid(row=2, column=0, sticky=N+S+E+W)

    varTFIDF_MNB = IntVar()
    c_MNB = Checkbutton(root, text="Multi-nominal Naive Bayes(TF-IDF)",
                        variable=varTFIDF_MNB).grid(row=3, column=1, sticky=W)
    btnMNB = Tkinter.ttk.Button(root, text="Multi-nominal Naive Bayes", command=lambda: MNB(
        X_train_t if varTFIDF_MNB.get() == 0 else X_train_tfidf, y_train, X_test_t if varTFIDF_MNB.get() == 0 else X_test_tfidf, y_test, ['1', '2'] if varBinMul.get() == 0 else ['1', '2', '3', '4', '5']), style='C.TButton').grid(row=3, column=0, sticky=N+S+E+W)

    varTFIDF_SGD = IntVar()
    c_SGD = Checkbutton(root, text="Stochastic Gradient Descent(TF-IDF)",
                        variable=varTFIDF_SGD).grid(row=4, column=1, sticky=W)
    btnSGD = Tkinter.ttk.Button(root, text="Stochastic Gradient Descent", command=lambda: SGDLog(
        X_train_t if varTFIDF_SGD.get() == 0 else X_train_tfidf, y_train, X_test_t if varTFIDF_SGD.get() == 0 else X_test_tfidf, y_test, ['1', '2'] if varBinMul.get() == 0 else ['1', '2', '3', '4', '5']), style='C.TButton').grid(row=4, column=0, sticky=N+S+E+W)

    varTFIDF_SVC = IntVar()
    c_SVC = Checkbutton(root, text="Support Vector Machine(TF-IDF)",
                        variable=varTFIDF_SVC).grid(row=5, column=1, sticky=W)
    btnSVC = Tkinter.ttk.Button(root, text="Support Vector Machine", command=lambda: SVCLinear(
        X_train_t if varTFIDF_SVC.get() == 0 else X_train_tfidf, y_train, X_test_t if varTFIDF_SVC.get() == 0 else X_test_tfidf, y_test, ['1', '2'] if varBinMul.get() == 0 else ['1', '2', '3', '4', '5']), style='C.TButton').grid(row=5, column=0, sticky=N+S+E+W)

    Tkinter.ttk.Label(
        root, text="Model Evaluation Options (With TF/TF-IDF)").grid(row=6, column=0, columnspan=2, sticky=N+S+E+W)

    Tkinter.ttk.Button(root, text="Model Analysis Binary(With TF)", command=lambda: modelAnalysis(CountVectorizer(max_features=1500, max_df=0.9,
                                                                                                                  stop_words=stopwords.words('english')), review[(review.stars == 1) | (review.stars == 5)]['text'], review[(review.stars == 1) | (review.stars == 5)]['stars']), style='C.TButton').grid(row=7, column=0, sticky=N+S+E+W)
    Tkinter.ttk.Button(root, text="Model Analysis Binary(With TF-IDF)", command=lambda: modelAnalysis(TfidfVectorizer(max_features=1500, max_df=0.9,
                                                                                                                      stop_words=stopwords.words('english')), review[(review.stars == 1) | (review.stars == 5)]['text'], review[(review.stars == 1) | (review.stars == 5)]['stars']), style='C.TButton').grid(row=7, column=1, sticky=N+S+E+W)
    Tkinter.ttk.Button(root, text="Model Analysis Multi-Class(With TF)", command=lambda: modelAnalysis(CountVectorizer(max_features=1500, max_df=0.9,
                                                                                                                       stop_words=stopwords.words('english')), review['text'], review['stars']), style='C.TButton').grid(row=8, column=0, sticky=N+S+E+W)
    Tkinter.ttk.Button(root, text="Model Analysis Multi-Class(With TF-IDF)", command=lambda: modelAnalysis(TfidfVectorizer(max_features=1500, max_df=0.9,
                                                                                                                           stop_words=stopwords.words('english')), review['text'], review['stars']), style='C.TButton').grid(row=8, column=1, sticky=N+S+E+W)
    root.style = Style()
    #('clam', 'alt', 'default', 'classic')
    style = Tkinter.ttk.Style()
    style.map("C.TButton",
              foreground=[('pressed', 'red'), ('active', "#00bcd4")],
              background=[('pressed', '!disabled', 'black'),
                          ('active', 'white')]
              )

    Tkinter.ttk.Style().configure("TButton", padding=10, relief="flat",
                                  background="#00bcd4", foreground="#616161")
    Tkinter.ttk.Style().configure("TCheckButton", padding=10, relief="flat",
                                  background="#00bcd4", foreground="#616161")
    Tkinter.ttk.Style().configure("TLabel", padding=10, relief="flat",
                                  background="#00bcd4", foreground="#ffffff")

    root.mainloop()


if __name__ == "__main__":
    main()
