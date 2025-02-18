import numpy as np
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Covariances
import os
import sys
import yaml
# sys.path.append((os.path.dirname(__file__)))

from src.clustering import KMeans_tW
from src.RemoteSensoringUtils import read_data
from src.hyperspectral_clustering_utils import (
        PCAImage, SlidingWindowVectorize,
        KmeansplusplusTransform, LabelsToImage,
        RemoveMeanImage, assign_segmentation_classes_to_gt_classes,
        compute_mIoU
)
from sklearn.pipeline import Pipeline
import time
import cloudpickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
        
    #config as dictionnary
    config = {"data_name":"salinas",#name of the dataset
              "dataset_path":"hyperspectral data",#path where data is already stored
              "small_dataset":False,#if true,a reduced dataset is used
              "reduce_factor":5,#reduction factor of dataset when 'small_dataset=True'
              "pca_dim":16,#number of selected features for PCA
              "window_size":11,#sliding window with overlap is used around each pixel for data sampling
              "n_init":5,#Number of initializations
              "max_iter":100,#Maximum number of iterations of t-Wishart clustering
              "seed":1234,#random seed
              "skip_zero_class":True,#Skip the zero class in the accuracy computation
              "kmeansplusplus_metric": "kullback",#kullback_sym #"kullback","kullbach_righr", "riemann"
              "n_jobs":-1,#Number of jobs to run in parallel
              "results_path":"",#Path to store results
              "show_plots":True,#Show plots (will block execution)
              "init":"kmeans++",#type of kmeans initialization, choose between "kmeans++" and "random"
              "tolerance":1e-4,#threshold for stopping Kmeans
              }
    exp=0
    path0= "results/hyperspectral clustering/"+config["data_name"]
    if not(os.path.exists(path0)):
        os.makedirs(path0)
    while os.path.exists(path0+f"/exp{exp}"):
        exp +=1
    os.makedirs(path0+f"/exp{exp}")
    config["results_path"]= path0+f"/exp{exp}"

# KMeans_tW: init 5/5 == kullback
# 100%|██████████| 16/16 [09:47<00:00, 36.72s/it]
# Kmeans++ init : Done! for seed 1880026316
# chosen centroids for init= [72319, 35473, 25968, 104859, 10153, 15511, 40576, 27819, 75407, 26013, 38774, 16409, 53868, 102132, 55410, 74665, 66589]
    
# runcell(0, 'C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/estimation journal paper/git code/src/hyperspectral_clustering_utils.py')
# runcell(0, 'C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/estimation journal paper/git code/src/RemoteSensoringUtils.py')
# runcell(0, 'C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/estimation journal paper/git code/src/manifold.py')
# runcell(0, 'C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/estimation journal paper/git code/src/tWishart.py')
# runcell(0, 'C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/estimation journal paper/git code/src/classification.py')
# runcell(0, 'C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/estimation journal paper/git code/src/clustering.py')
# runcell(0, 'C:/Users/Imen Ayadi/OneDrive - CentraleSupelec/Bureau/thèse/estimation journal paper/git code/scripts/hyperspectral_clustering.py')

    # Create results folder
    os.makedirs(config["results_path"], exist_ok=True)
    
    with open(os.path.join(config["results_path"], "config.yml"), "w") as f:
        yaml.dump(config, f)

   
    # Read data
    print('1) Reading data...')
    data, labels, labels_name = read_data(config['dataset_path'], config['data_name'])
    n_classes = len(labels_name)

    pipeline_baseline = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('scm', Covariances('scm')),
        ('kmeans', KmeansplusplusTransform(
            n_clusters=n_classes, n_jobs=config['n_jobs'], n_init=config['n_init'],
            random_state=config['seed'],tol=config['tolerance'],
            use_plusplus=(config["init"]=="kmeans++"), 
            verbose=1, max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)


    pipeline_tW_df10 = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_tW(n_clusters=n_classes, 
                             dfs = [10],
                             init=config["init"],
                             kmeansplusplus_metric=config["kmeansplusplus_metric"],
                             n_jobs=config['n_jobs'],tol=config['tolerance'],
                            random_state=config['seed'], 
                            n_init=config['n_init'], verbose=1, 
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)
    
    
    pipeline_W = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_tW(n_clusters=n_classes, 
                             dfs = [np.inf],
                             init=config["init"],
                             n_jobs=config['n_jobs'],tol=config['tolerance'],
                            random_state=config['seed'], 
                            n_init=config['n_init'], verbose=1, 
                            kmeansplusplus_metric=config["kmeansplusplus_metric"],
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)

    pipeline_tW_kurtosis = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_tW(n_clusters=n_classes, dfs = None,init=config["init"],
                              df_estimation_method="kurtosis estimation",
                              n_jobs=config['n_jobs'],rmt=True,
                              kmeansplusplus_metric=config["kmeansplusplus_metric"],
                            random_state=config['seed'], tol=config['tolerance'],
                            n_init=config['n_init'], verbose=1, 
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)
    
    pipeline_tW_pop = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_tW(n_clusters=n_classes, dfs = None,init=config["init"],
                              df_estimation_method="pop exact",
                              n_jobs=config['n_jobs'],rmt=True,
                              kmeansplusplus_metric=config["kmeansplusplus_metric"],
                            random_state=config['seed'], tol=config['tolerance'],
                            n_init=config['n_init'], verbose=1, 
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)
    
    pipeline_tW_pop_approx = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_tW(n_clusters=n_classes, dfs = None,init=config["init"],
                              df_estimation_method="pop approx",
                              n_jobs=config['n_jobs'],rmt=True,
                              kmeansplusplus_metric=config["kmeansplusplus_metric"],
                            random_state=config['seed'], tol=config['tolerance'],
                            n_init=config['n_init'], verbose=1, 
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)
    
    
    pipeline_tW_notpop = Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_tW(n_clusters=n_classes, dfs = None,init=config["init"],
                              df_estimation_method="notpop",
                              n_jobs=config['n_jobs'],rmt=False,
                              kmeansplusplus_metric=config["kmeansplusplus_metric"],
                            random_state=config['seed'], tol=config['tolerance'],
                            n_init=config['n_init'], verbose=1, 
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)
    
    pipeline_tW_sh= Pipeline([
        ('remove_mean', RemoveMeanImage()),
        ('pca', PCAImage(n_components=config['pca_dim'])),
        ('sliding_window', SlidingWindowVectorize(config['window_size'])),
        ('kmeans', KMeans_tW(n_clusters=n_classes, dfs = None,init=config["init"],
                              df_estimation_method="shrinkage",
                              n_jobs=config['n_jobs'],rmt=False,
                              kmeansplusplus_metric=config["kmeansplusplus_metric"],
                            random_state=config['seed'], tol=config['tolerance'],
                            n_init=config['n_init'], verbose=1, 
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)
    
    pipelines = {
            "tW-Kmeans-df=10": pipeline_tW_df10,
            "W-Kmeans": pipeline_W,
            "tW-Kmeans-df-kurtosis": pipeline_tW_kurtosis,
            #"tW-Kmeans-df-pop": pipeline_tW_pop,
            #"tW-Kmeans-df-pop-approx": pipeline_tW_pop_approx,
            "tW-Kmeans-df-notpop": pipeline_tW_notpop,
            #"tW-Kmeans-df-shrinkage": pipeline_tW_sh,
            "baseline": pipeline_baseline,


    }
    
    results = {}
    durations = {}#duration of the whole pipeline
    images = {}
    accuracies = {}
    ious = {}
    mious = {}
    dfs = {}
    
    for name, pipeline in pipelines.items():
        
        print(f'2) Fitting and transforming {name}...')
        start = time.time()
        result = pipeline.fit_transform(data, labels)
        end = time.time()
        results[name] = result
        durations[name] = end - start

        print(f'Computing accuracy for {name}...')

        # Padding to take into account the window size
        result_final = np.ones_like(data[:, :, 0])*np.nan
        pad_x = (data.shape[0] - result.shape[0])//2
        pad_y = (data.shape[1] - result.shape[1])//2
        result_final[pad_x:-pad_x, pad_y:-pad_y] = result

        # Match labels 
        result_final = assign_segmentation_classes_to_gt_classes(result_final, labels)

        # Replace class 0 with ground truth since no data is available
        if config['skip_zero_class']:
            mask = np.logical_and(np.logical_not(np.isnan(result_final)), labels==0)
            result_final[mask] = 0
            mask = np.logical_and(labels!=0, ~np.isnan(result_final))
        else:
            mask = ~np.isnan(result_final)
        accuracy = accuracy_score(labels[mask],
                                result_final[mask])
        print(f'Overall accuracy {name}: {accuracy}')
        ious[name], mious[name] = compute_mIoU(result_final, labels)
        print(f'mIoU {name}: {mious[name]}')
        print(f'IoU {name}: {ious[name]}')

        images[name] = result_final
        accuracies[name] = accuracy
       
        if "tW" in name:
            dfs[name] = pipeline.named_steps["kmeans"].dfs                
            
       

    # Store results
    toStore = {
            'config': config,
            'pipelines': pipelines,
            'results': results,
            'durations' : durations,
            'images': images,
            'accuracies': accuracies,
            'ious': ious,
            'mious': mious,
            'pipelines': pipelines,
            'dfs':dfs,
        }
    
    if config['results_path'] is not None:
        with open(os.path.join(config['results_path'], 'results.pkl'), 'wb') as f:
            cloudpickle.dump(toStore, f)

    # Show results
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(labels, aspect='auto', cmap='tab20')
    plt.title('Ground truth')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(config['results_path'], 'gt.pdf'))

    for i, name in enumerate(pipelines.keys()):
        fig = plt.figure(figsize=(13, 5))
        plt.imshow(images[name], aspect='auto', cmap='tab20')
        if name in accuracies:
            accuracy = accuracies[name]
            mIoU = mious[name]
            iou = ious[name]
            print(name, accuracy, mIoU)
            plt.title(f'{name} (acc={accuracy:.4f}, mIoU={mIoU:.4f}')
        else:
            plt.title(f'{name}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(config['results_path'], f'{name}.pdf'))
    
    if config['show_plots']:
        plt.show()
