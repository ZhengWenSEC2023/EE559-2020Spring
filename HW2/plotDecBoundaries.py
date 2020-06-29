################################################
## EE559 HW Wk2, Prof. Jenkins, Spring 2018
## Created by Arindam Jati, TA
## Tested in Python 3.6.3, OSX El Captain
################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from knn import OVRClassfierPlot
import matplotlib.patches as mpatches

def plotDecBoundariesDouble(training, label_train, sample_mean):
    # Plot the decision boundaries and data points for minimum distance to
    # class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # sample_mean: mean vector for each class
    #
    # Total number of classes
    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    
    inc = 0.005
    
    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                         np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                    y.reshape(y.shape[0] * y.shape[1], 1, order='F')))  
    
    # make (x,y) pairs as a bunch of row vectors.
    # distance measure evaluations for each (x,y) pair.
    
    dist_mat = cdist(xy, sample_mean)
    pred_label = np.argmin(dist_mat, axis=1)

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    #
    # show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    # plot the class training data.
    if (np.unique(label_train) == np.array([1,2])).all():
        plt.plot(training[label_train == 1, 0], training[label_train == 1, 1], 'rx')
        plt.plot(training[label_train == 2, 0], training[label_train == 2, 1], 'go')
    elif (np.unique(label_train) == np.array([1,3])).all():
        plt.plot(training[label_train == 1, 0], training[label_train == 1, 1], 'rx')
        plt.plot(training[label_train == 3, 0], training[label_train == 3, 1], 'b*')
    else:
        plt.plot(training[label_train == 2, 0], training[label_train == 2, 1], 'go')
        plt.plot(training[label_train == 3, 0], training[label_train == 3, 1], 'b*')
        
    # include legend for training data
    if (np.unique(label_train) == np.array([1,2])).all():
        l = plt.legend(('Class 1', 'Class 2&3'), loc=2)
    elif (np.unique(label_train) == np.array([1,3])).all():
        l = plt.legend(('Class 1&2', 'Class 3'), loc=2)
    else:
        l = plt.legend(('Class 2', 'Class 1&3'), loc=2)
    plt.gca().add_artist(l)

    # plot the class mean vector.
    if (np.unique(label_train) == np.array([1,2])).all():
        m1, = plt.plot(sample_mean[0, 0], sample_mean[0, 1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
        m2, = plt.plot(sample_mean[1, 0], sample_mean[1, 1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    elif (np.unique(label_train) == np.array([1,3])).all():
        m1, = plt.plot(sample_mean[0, 0], sample_mean[0, 1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
        m2, = plt.plot(sample_mean[1, 0], sample_mean[1, 1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')
    else:
        m1, = plt.plot(sample_mean[0, 0], sample_mean[0, 1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
        m2, = plt.plot(sample_mean[1, 0], sample_mean[1, 1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

    # include legend for class mean vector
    if (np.unique(label_train) == np.array([1,2])).all():
        l1 = plt.legend([m1, m2], ['Class 1 Mean', 'Class 2&3 Mean'], loc=4)
    elif (np.unique(label_train) == np.array([1,3])).all():
        l1 = plt.legend([m1, m2], ['Class 1&2 Mean', 'Class 3 Mean'], loc=4)
    else:
        l1 = plt.legend([m1, m2], ['Class 2 Mean', 'Class 1&3 Mean'], loc=4)

    plt.gca().add_artist(l1)

    plt.show()


def plotDecBoundariesMul(train_set):
    # Plot the decision boundaries and data points for minimum distance to
    # class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # sample_mean: mean vector for each class
    #
    # Total number of classes
    # Set the feature range for ploting
    training = train_set[:, 0:-1]
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1
    label_train = train_set[:, -1]

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.01

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                         np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                    y.reshape(y.shape[0] * y.shape[1], 1, order='F')))  
    
    # make (x,y) pairs as a bunch of row vectors.
    # distance measure evaluations for each (x,y) pair.
    pred_label = OVRClassfierPlot(train_set, xy)

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    
    values = np.unique(decisionmap.ravel())
    
    # show the image, give each coordinate a color according to its class label
    im = plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[0], label="Undecided"), 
               mpatches.Patch(color=colors[1], label="Class 1"), 
               mpatches.Patch(color=colors[2], label="Class 2"), 
               mpatches.Patch(color=colors[3], label="Class 3")
               ]
    l1 = plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # plot the class training data.
    plt.plot(training[label_train == 1, 0], training[label_train == 1, 1], 'rx')
    plt.plot(training[label_train == 2, 0], training[label_train == 2, 1], 'go')
    plt.plot(training[label_train == 3, 0], training[label_train == 3, 1], 'b*')
    plt.legend(('Class 1', 'Class 2','Class 3'), loc=2)
    
    
    plt.gca().add_artist(l1)
    # include legend for training data
    # plot the class mean vector.

    plt.show()
    

def plotDecBoundariesMVM(means):
    # Plot the decision boundaries and data points for minimum distance to
    # class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # sample_mean: mean vector for each class
    #
    # Total number of classes
    # Set the feature range for ploting

    xrange = (-3, 3)
    yrange = (-3, 3)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.01

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                         np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                    y.reshape(y.shape[0] * y.shape[1], 1, order='F')))  
    
    dist_mat = cdist(xy, means)

    # make (x,y) pairs as a bunch of row vectors.
    # distance measure evaluations for each (x,y) pair.
    pred_label = np.argmin(dist_mat, axis=1)

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    
    values = np.unique(decisionmap.ravel())
    
    # show the image, give each coordinate a color according to its class label
    im = plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
    colors = [ im.cmap(im.norm(value)) for value in values]
    if np.shape(colors)[0] == 2:
        patches = [mpatches.Patch(color=colors[0], label="Gamma 1"), 
                   mpatches.Patch(color=colors[1], label="Gamma 2")]
    else:
        patches = [mpatches.Patch(color=colors[0], label="Gamma 1"), 
                   mpatches.Patch(color=colors[1], label="Gamma 2"),
                   mpatches.Patch(color=colors[2], label="Gamma 3")]
            
    l1 = plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # plot the class training data.
    m1, = plt.plot(means[0, 0], means[0, 1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(means[1, 0], means[1, 1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    if np.shape(colors)[0] != 2:
        m3, = plt.plot(means[2, 0], means[2, 1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')
        
    if np.shape(colors)[0] == 2:
        l2 = plt.legend([m1, m2], ['mu1', 'mu2'], loc=4)
    else:
        l2 = plt.legend([m1, m2, m3], ['mu1', 'mu2', 'mu3'], loc=4)

    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)

    # include legend for training data
    # plot the class mean vector.

    plt.show()