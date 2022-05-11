from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle

from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts

from nearest_neighbor_classify import nearest_neighbor_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Step 0: Set up

#You will need to report performance for two
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier

#The starter code does not crash when run unmodified 
#and you can get a preview of how results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='bag_of_sift')
parser.add_argument('--classifier', help='classifier', type=str, default='nearest_neighbor')
parser.add_argument('--dataset_path', help='dataset path', type=str, default='./p1_data/p1/')
args = parser.parse_args()

DATA_PATH = args.dataset_path

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
              'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
 
ABBR_CATEGORIES = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as well.
NUM_TRAIN_PER_CAT = 100


FEATURE = args.feature
CLASSIFIER = args.classifier

def main(): 
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    

    # TODO Step 1:
    # Represent each image with the appropriate feature
    # Each function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the 
    # dimensionality of each image representation. See the starter code for
    # each function for more details.

    if FEATURE == 'tiny_image':
        # TODO Modify get_tiny_images.py 
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)

    elif FEATURE == 'bag_of_sift':
        # TODO Modify build_vocabulary.py
        if os.path.isfile('vocab.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')
            vocab_size = 230   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to compute
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile('train_image_feats.pkl') is False:
            # TODO Modify get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths)
            with open('train_image_feats.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths)
            with open('test_image_feats.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
    
    elif FEATURE == 'dumy_feature':
        train_image_feats = []
        test_image_feats = []
    
    else:
        raise NameError('Unknown feature type')

    # TODO Step 2: 
    # Classify each test image by training and using the appropriate classifier
    # Each function to classify test features will return an N x 1 array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels.

    if CLASSIFIER == 'nearest_neighbor':
        # TODO Modify nearest_neighbor_classify.py
        predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
    
    elif CLASSIFIER == 'dumy_classifier':
        # The dummy classifier simply predicts a random category for every test case
        predicted_categories = test_labels[:]
        shuffle(predicted_categories)
    
    else:
        raise NameError('Unknown classifier type')

    accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
    # Step 3: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section. 
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')
    plt.savefig("{}.png".format(FEATURE))
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_image_paths(data_path, categories, num_train_per_cat):
    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []

    for category in categories:

        image_paths = glob(os.path.join(data_path, 'train', category, '*.jpg'))
        for i in range(num_train_per_cat):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)

        image_paths = glob(os.path.join(data_path, 'test', category, '*.jpg'))
        for i in range(num_train_per_cat):
            test_image_paths.append(image_paths[i])
            test_labels.append(category)

    return train_image_paths, test_image_paths, train_labels, test_labels


if __name__ == '__main__':
    main()

