import numpy as np
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Covariances
import os
import sys
import yaml
sys.path.append((os.path.dirname(__file__)))

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
    config = {"data_name":"indianpines",#name of the dataset
              "dataset_path":"hyperspectral data",#path where data is already stored
              "small_dataset":False,#if true,a reduced dataset is used
              "reduce_factor":5,#reduction factor of dataset when 'small_dataset=True'
              "pca_dim":5,#number of selected features for PCA
              "window_size":5,#sliding window with overlap is used around each pixel for data sampling
              "n_init":10,#Number of initializations
              "max_iter":100,#Maximum number of iterations of -Wishart clustering
              "seed":1234,#random seed
              "skip_zero_class":True,#Skip the zero class in the accuracy computation
              "n_jobs":-1,#Number of jobs to run in parallel
              "results_path":"results/hyperspectral clustering/indianpines",#Path to store results
              "show_plots":True,#Show plots (will block execution)
              "init":"random",#type of kmeans initialization, choose between "kmeans++" and 
              "tolerance":1e-4,#threshold for stopping Kmeans
              }

    

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
            use_plusplus=(config["init"]=="kmeans++"), verbose=1, max_iter=config['max_iter'])),
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
                            max_iter=config['max_iter'])),
        ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
                                        config['window_size']))
        ], verbose=True)

    # pipeline_tW_df_simple = Pipeline([
    #     ('remove_mean', RemoveMeanImage()),
    #     ('pca', PCAImage(n_components=config['pca_dim'])),
    #     ('sliding_window', SlidingWindowVectorize(config['window_size'])),
    #     ('kmeans', KMeans_tW(n_clusters=n_classes, dfs = None,init=config["init"],
    #                          df_estimation_method="kurtosis_estimation",
    #                          n_jobs=config['n_jobs'],
    #                         random_state=config['seed'], tol=config['tolerance'],
    #                         n_init=config['n_init'], verbose=1, 
    #                         max_iter=config['max_iter'])),
    #     ('labels_to_image', LabelsToImage(data.shape[0], data.shape[1],
    #                                     config['window_size']))
    #     ], verbose=True)

    
    
    pipelines = {
            "baseline": pipeline_baseline,
            "tW-Kmeans-df=10": pipeline_tW_df10,
            "W-Kmeans": pipeline_W,
            #"tW-Kmeans-df-simple": pipeline_tW_df_simple,

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
            plt.title(f'{name} (acc={accuracy:.2f}, mIoU={mIoU:.2f}')
        else:
            plt.title(f'{name}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(config['results_path'], f'{name}.pdf'))
    
    if config['show_plots']:
        plt.show()

