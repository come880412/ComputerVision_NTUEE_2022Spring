import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    train_labels = np.array(train_labels)
    test_predicts = []
    k = 5

    distance_to_train_feats = distance.cdist(test_image_feats, train_image_feats, 'braycurtis')
    distance_index_sort = np.argsort(distance_to_train_feats, axis=1)
    top_k_label = train_labels[distance_index_sort[:, :k]]

    for k_label in top_k_label:
        unique, pos = np.unique(k_label, return_inverse=True) #Finds all unique elements and their positions
        counts = np.bincount(pos)
        if len(counts) == k:
            random_index = np.random.choice(k, 1)[0]
            test_predicts.append(unique[random_index])
        else:
            maxpos = counts.argmax()
            test_predicts.append(unique[maxpos])
    

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
