import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from make_arff import load_arff, get_sampled_sorted_word_list
from sklearn import metrics
import numpy as np
import itertools
import pickle

ARFF_PATH = 'data.arff'

data, meta = load_arff(ARFF_PATH) 
authors = list(meta['_author_'][1])

def preprocess():
    li_data = data.tolist()
    print(authors)

    feature_data = [ np.array(x[1:], dtype=int) for x in li_data ]
    targets = [ np.int(authors.index(x[0].decode())) for x in li_data ]  # enumerate this
    return train_test_split(feature_data, targets, test_size=0.1)

def fit(model, x_train, y_train, kwargs):
    clf = model(**kwargs).fit(x_train, y_train)
    return clf

def get_top_n_features(clf, n=10):
    sampled_sorted_words = pickle.load(open('wordcount.pylist', 'rb'))
    for i in range(0, clf.best_estimator_.coef_.shape[0]):
        top10 = np.argsort(clf.best_estimator_.coef_[i])[-n:]
    top10_words = [sampled_sorted_words[x] for x in top10]
    print('Top %d Features: ' % n, top10_words)
    return top10_words

def classify(x_train, x_test, y_train, y_test):
    param_ranges = [1/10**a  for a in range(1, 11)] + [0]
    tuned_parameters = {'alpha': param_ranges}

    model = BernoulliNB()
    grid_search = GridSearchCV(
                    model,
                    tuned_parameters, 
                    verbose=True,
                    n_jobs=-1,
                    cv=10, 
                    scoring='f1_macro'
                  )
    clf = grid_search.fit(x_train, y_train)

    print(clf.__class__.__name__)
    print(model.__class__.__name__)
    print('Parameters: ', clf.best_params_)

    get_top_n_features(clf, n=25)

    predictions = clf.predict(x_test)
    report = metrics.classification_report(y_test, predictions, target_names=authors)

    accuracy = metrics.accuracy_score(y_test, predictions)
    confusion = metrics.confusion_matrix(y_test, predictions)

    print(report)
    print('Accuracy: ', accuracy)
    print('Confusion Matrix: ')
    print(confusion)
    plot_confusion_matrix(confusion, authors)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    data = preprocess()
    classify(*data)

if __name__ == '__main__':
    main()

