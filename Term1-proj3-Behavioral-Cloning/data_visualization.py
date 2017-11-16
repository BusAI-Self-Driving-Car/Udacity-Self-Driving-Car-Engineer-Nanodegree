### Data exploration visualization code
import random
import matplotlib.pyplot as plt

## Visualizations will be shown in the notebook.
#%matplotlib inline

## Visualizations will be shown in the notebook.
#get_ipython().run_line_magic('matplotlib', 'inline')


def visualize_images(X, y, num_random_imgs=10, color_map=None):
    random.seed(2342)
    
    fig, axs = plt.subplots(int(num_random_imgs/4), 4, figsize=(15, 15))
    fig.subplots_adjust(hspace = .4, wspace=.003)
    axs = axs.ravel()
    for i in range(0, num_random_imgs, 2):
        index = random.randint(0, len(X)) - 1
        if index%2!=0:
            index+=1 # We want an even index
        
        if color_map=='gray':
            image = X[index].squeeze()
            image_flipped = X[index+1].squeeze()
        else:
            image = X[index]
            image_flipped = X[index+1]
            #print("image shape = {}".format(image_flipped.shape))
                
        axs[i].axis('off')
        axs[i].imshow(image, color_map)
        axs[i].set_title("{} steering angle {}".format(index, y[index]))
        
        axs[i+1].axis('off')
        axs[i+1].imshow(image_flipped, color_map)
        axs[i+1].set_title("{0} steering angle {1:.3f}".format(index+1, y[index+1]))

def plot_histogram(data):    
    len_data = len(data)
    print("length of data = {}".format(len_data))
    num_bins=int(2*len_data/400)
    print("no. of bins = {}".format(num_bins))
    print()
    
    plt.hist(data, bins=num_bins)
    # plt.xlim(xmax=n_classes-1)
    plt.ylabel('Number of Occurences')
    plt.xlabel('Steering angle')
    plt.title('Distribution of steering angles in the dataset')