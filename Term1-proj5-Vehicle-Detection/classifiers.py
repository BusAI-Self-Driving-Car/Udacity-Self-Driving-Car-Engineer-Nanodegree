
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML

### Widen notebook to fit browser window
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


get_ipython().run_cell_magic('javascript', '', '// Press ctrl-M followed by r to "run all cells" in this notebook\n\nJupyter.keyboard_manager.command_shortcuts.add_shortcut(\'r\', {\n    help : \'run all cells\',\n    help_index : \'zz\',\n    handler : function (event) {\n        IPython.notebook.execute_all_cells();\n        return false;\n    }}\n);')


# In[3]:


import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from file_utils import get_img_filenames
from configuration import dict_config_params
from feature_extracters import extract_features


# In[4]:


def fit_svm(X, labels, verbose=True):
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, labels, 
                                                        test_size=0.2, 
                                                        random_state=rand_state) 
    
    # Use a linear SVC 
    svc = LinearSVC()
    
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    if verbose:
        print("\n",round(t2-t, 2), 'Seconds to train SVC...')
        print('\nTest Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        
        t=time.time()    
        n_predict = 10
        print('\nMy SVC predicts:     ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print("\n",round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    return svc, X_scaler


# In[5]:


def get_training_data():
    
    img_filenames_cars, img_filenames_notcars = get_img_filenames()
    
    truncate_data_at = dict_config_params['truncate_training_data_after']
    
    hog_feat = dict_config_params['use_hog_feat']
    spatial_feat = dict_config_params['use_spatial_feat']
    hist_feat = dict_config_params['use_hist_feat']
    
    ## Feature vectors CAR
    feature_vectors_car = []
    count = 1
    for img_filename in img_filenames_cars:    
        image = mpimg.imread(img_filename)

        img_features = extract_features(image, verbose=False, 
                                    hog_feat=hog_feat, spatial_feat=spatial_feat, hist_feat=hist_feat)
        feature_vectors_car.append(img_features)
        
        if truncate_data_at > 0 and count > truncate_data_at:
            break
        count += 1

    ## Feature vectors NOT-CAR
    feature_vectors_notcar = []
    count = 1
    for img_filename in img_filenames_notcars:    
        image = mpimg.imread(img_filename)

        img_features = extract_features(image, verbose=False, 
                                    hog_feat=hog_feat, spatial_feat=spatial_feat, hist_feat=hist_feat)
        feature_vectors_notcar.append(img_features)
        
        if truncate_data_at > 0 and count > truncate_data_at:
            break
        count += 1


    # Create an array stack of feature vectors
    X = np.vstack((feature_vectors_car, feature_vectors_notcar)).astype(np.float64)                        
    print("len(X): {}".format(len(X)))

    # Define the labels vector
    labels = np.hstack((np.ones(len(feature_vectors_car)), 
                        np.zeros(len(feature_vectors_notcar))))
    
    return X, labels

