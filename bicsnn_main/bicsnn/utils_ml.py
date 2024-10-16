import glob
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sklearn.metrics as m
from sklearn.decomposition import PCA
from sklearn import manifold

# import SVC classifier
from sklearn.svm import LinearSVC
import seaborn as sns

from bicsnn.output.figures import spiderplot_from_mat, barplot_from_mat

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 4

def read_spikes(folder_path):
    """Reads all pt files from a folder in name order.
    """
    pt_files = []
    for filename in sorted(glob.glob(os.path.join(folder_path, "*.pkl"))):
        pt_files.append(filename)
    return pt_files

def extract_spikes(files):
    first = True
    # Everyfile is a batch of spikes from different
    for filename in files:
        with open(filename, 'rb') as handle:
            batch_dict = pickle.load(handle)
        if first:
            spikes, targets = batch_dict['spikes'], batch_dict['targets']
            first = False
        else:
            spikes = np.hstack((spikes, batch_dict['spikes']))
            targets = np.hstack((targets, batch_dict['targets']))
    return spikes, targets


def pca_figures(spikes, labels, prefix=''):
    # Compress time of spikes into sum
    data = np.sum(spikes, axis=0)
    # PCA to reduce the dimensionality of the data before putting it into the classifier
    pca = PCA(n_components=data.shape[1])
    data_pca = pca.fit_transform(data)
    #print('Explained variance ratio /n', pca.explained_variance_ratio_)
    exp_var = np.cumsum(pca.explained_variance_ratio_)
    print("Variance explained 70%-{}, 80%-{}, 90%-{}".format(np.sum(exp_var<0.70), np.sum(exp_var<0.80), np.sum(exp_var<0.90)))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(exp_var, 'o-')
    #sns.set(font_scale=0.4)  # font size 2
    ax1.set_ylabel("Variance explained")
    ax1.set_xlabel("Number of components")
    #plt.tight_layout()
    #plt.savefig('./figures/benchmark/pca_variance_explained.png', dpi=300)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    # print('Explained variance ratio /n', pca.explained_variance_ratio_)
    for label in np.unique(labels):
        data_class = data_pca[labels==label]
        ax2.plot(data_class[:, 0], data_class[:, 1])
    ax2.set_ylabel("PCA 2")
    ax2.set_xlabel("PCA 1")
    ax2.legend(np.unique(labels))
    plt.tight_layout()
    plt.savefig('./figures/benchmark/{}_pca_2D.png'.format(prefix), dpi=300)

    # Manifolds for representation
    sr_lle, sr_err = manifold.locally_linear_embedding(data, n_neighbors=12, n_components=3)
    sr_tsne = manifold.TSNE(n_components=3, perplexity=40, random_state=0).fit_transform(data)

    fig = plt.figure(figsize=(8, 6))
    ax0 = fig.add_subplot(211, projection="3d")
    ax1 = fig.add_subplot(212, projection="3d")

    ax0.scatter(sr_lle[:, 0], sr_lle[:, 1], sr_lle[:, 2], c=labels)
    ax0.set_title("LLE Embedding")
    ax1.scatter(sr_tsne[:, 0], sr_tsne[:, 1], sr_tsne[:, 2], c=labels)
    _ = ax1.set_title("t-SNE Embedding")
    #plt.show()
    plt.savefig('./figures/benchmark/{}_manifolds.png'.format(prefix), dpi=300)


def svm_analysis(spikes, labels, prefix='', folder_fig=None, Tune_hyp = False):

    # Train/Test given already
    if isinstance(spikes, list):
        classes = np.unique(labels[0])
        n_classes = len(classes)
        X_train, X_test, y_train, y_test = spikes[0], spikes[1], labels[0], labels[1]

    else:
        # Create training and testing sets
        classes = np.unique(labels)
        n_classes = len(classes)
        # Compress time of spikes into sum
        data =np.sum(spikes, axis=0)
        #print('Number of classes in the dataset %s' % n_classes)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,
                                                            shuffle=True)  # 0.3 train percent   random_state=42

    if Tune_hyp:
        # Hyperparameterisation Params of model
        C_range = np.logspace(-2, 10, 12)
        tol_range = np.logspace(-9, 3, 12)
        param_grid = dict(tol=tol_range, C=C_range)
        svc = LinearSVC()  # SVC(kernel='rbf', C=Cval, gamma=gamma) |SVC(kernel='linear', C=Cval) | SVC(kernel='poly', C=Cval)

        cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
        grid = GridSearchCV(svc, param_grid=param_grid, cv=cv)
        grid.fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        C = grid.best_params_['C']
        tol = grid.best_params_['tol']
    else:
        # Last found with test 30% and 5folds linear
        C = 0.01
        tol = 0.00000187
        # rbf 30% 5fold
        # 'C': 1000000.0, 'gamma': 1e-05}

    # Train and testing
    svc = LinearSVC(C=C, tol=tol) # Deafult : l2-penalty ovr
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    y_pred_train = svc.predict(X_train)
    test_acc = m.balanced_accuracy_score(y_test, y_pred)
    train_acc = m.balanced_accuracy_score(y_train, y_pred_train)
    print('Model accuracy score with linear kernel : {0:0.4f}'.format(test_acc))
    print('Training-set score: {0:0.4f} Overfit?'.format(train_acc))



    if folder_fig is not None:
        # Spider plot from coefficients of the Linear Classifier
        spiderplot_from_mat(svc.coef_, folder_fig)
        barplot_from_mat(svc.coef_, folder_fig)

        # Confusion Matrix and Viualization
        cm = m.multilabel_confusion_matrix(y_test, y_pred, labels=range(n_classes))
        plt.figure(dpi=300)
        plt.ioff()
        for cl in range(n_classes):
            # print('Confusion matrix of class ', Letter_written[cl], '\n', cm[cl])  # The values are counter-intuitive with the classical definition
            # print('True Positives(TP) = ', cm[cl][1, 1])
            # print('True Negatives(TN) = ', cm[cl][0, 0])
            # print('False Positives(FP) = ', cm[cl][0, 1])
            # print('False Negatives(FN) = ', cm[cl][1, 0], '\n')

            plt.subplot(4, 3, cl + 1)
            plt.title("L {}".format(classes[cl]))
            cm_matrixi = pd.DataFrame(data=[[cm[cl][1, 1], cm[cl][0, 1]], [cm[cl][1, 0], cm[cl][0, 0]]],
                                      columns=['A+', 'A-'],
                                      index=['P+', 'P-'])
            sns.heatmap(cm_matrixi, annot=True, fmt='d', cmap='YlGnBu', vmin=0, vmax=2)
        sns.set(font_scale=0.4)  # font size 2
        plt.tight_layout()

        plt.savefig(folder_fig.joinpath('{}_confusion_matrix.png'.format(prefix)), format='png', dpi=300)
        plt.close()

    return test_acc
