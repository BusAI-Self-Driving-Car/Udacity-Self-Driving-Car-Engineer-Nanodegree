# coding: utf-8

from os import listdir
from os.path import isfile, join

import glob

def list_img_filenames(imgs_dir):
    # Read in filenames from all subdirs
    img_filenames = []
    for directory in listdir(imgs_dir):
            
        dir_full_path = join(imgs_dir, directory)
        
        #print()
        #print("dir_full_path: {}".format(dir_full_path))
    
        img_filenames.append(glob.glob(dir_full_path+"/*.png"))
        
    img_filenames = [item for sublist in img_filenames for item in sublist]
    return img_filenames


def get_img_filenames():
    # Read in our vehicles and non-vehicles
    imgs_dir = '../../data/vehicle_detection/'
    imgs_dir_cars = join(imgs_dir, 'vehicles/')
    imgs_dir_notcars = join(imgs_dir, 'non-vehicles/')
    
    print("listdir(imgs_dir_cars): {}".format(listdir(imgs_dir_cars)))
    print("listdir(imgs_dir_notcars): {}".format(listdir(imgs_dir_notcars)))
    
    # Import filenames of all "car" images in the dataset
    img_filenames_cars = list_img_filenames(imgs_dir_cars)
    print()
    print("len(img_filenames_cars): {}".format(len(img_filenames_cars)))
    
    # Import filenames of all "notcar" images in the dataset
    img_filenames_notcars = list_img_filenames(imgs_dir_notcars)
    print("len(img_filenames_notcars): {}".format(len(img_filenames_notcars)))
    
    return img_filenames_cars, img_filenames_notcars