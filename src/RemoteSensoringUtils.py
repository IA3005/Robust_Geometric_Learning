import numpy as np
from scipy.io import loadmat
from typing import Tuple, Dict
from numpy import ndarray
import os
import wget


def download_hyperspectral_data(data_path,dataset_name):
    """Download hyperspectral dataset from the web."""
    datasets = {
        'indianpines': [
            'https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat',
            'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
            'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
        ],
        'salinas': [
            'https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat',
            'https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
            'https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
            ],
        'Pavia': [
            'https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
            'https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'
            ],
        'PaviaU': [
            'https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
            'https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'
            ],
        'KSC': [
            'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
            'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'
            ]
        }

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    if dataset_name in datasets.keys():
        print(f"Downloading {dataset_name} dataset...")
        if not os.path.exists(data_path+f"/{dataset_name}"):
            print(f"Creating {dataset_name} directory...")
            os.makedirs(data_path+f"/{dataset_name}")
        try:
            for url in datasets[dataset_name]:
                filename = url.split("/")[-1]
                if not os.path.exists(data_path+f"/{dataset_name}/{filename}"):
                    print(f"Downloading {url}...")
                    wget.download(url, out=data_path+f"/{dataset_name}")
                    print("\n")
                else:
                    print(f"File {filename} already exists. Skipping...")
        except Exception as e:
            print(f"Error downloading {dataset_name} dataset:")
            print(f"{e}")
    else:
        print("Wrong dataset name!")
    return None


def read_salinas(data_path: str, version: str="corrected") -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    """
    Read Salinas hyperspectral data.

    Parameters
        data_path: str
            Path to the data folder
        version: str
            Version of the data to read. Can be either "corrected" or "raw".
            Default is "corrected".


    Returns
        data: ndarray
            Data array of shape (512, 217, 204).
        labels: ndarray
            Labels array of shape (512, 217).
        labels_names: Dict[int, str]
            Dictionary mapping labels to their names.
    """
    if version == "corrected":
        data_file = os.path.join(data_path, "Salinas_corrected.mat")
    else:
        data_file = os.path.join(data_path, "Salinas.mat")
    data = loadmat(data_file)['salinas_corrected']
    labels = loadmat(os.path.join(data_path,
                                  'Salinas_gt.mat'))['salinas_gt']
    labels_names = {
                0: 'Undefined',
                1: 'Brocoli_green_weeds_1',
                2: 'Brocoli_green_weeds_2',
                3: 'Fallow',
                4: 'Fallow_rough_plow',
                5: 'Fallow_smooth',
                6: 'Stubble',
                7: 'Celery',
                8: 'Grapes_untrained',
                9: 'Soil_vinyard_develop',
                10: 'Corn_senesced_green_weeds',
                11: 'Lettuce_romaine_4wk',
                12: 'Lettuce_romaine_5wk',
                13: 'Lettuce_romaine_6wk',
                14: 'Lettuce_romaine_7wk',
                15: 'Vinyard_untrained',
                16: 'Vinyard_vertical_trellis'
            }
    return data, labels, labels_names


def read_indian_pines(data_path: str, version: str="corrected") -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    """
    Read Indian Pines hyperspectral data.
    Parameters
        data_path: str
            Path to the data folder
        version: str
            Version of the data to read. Can be either "corrected" or "raw".
            Default is "corrected".

    Returns
        data: ndarray
            Data array of shape (145, 145, 200).
        labels: ndarray
            Labels array of shape (145, 145).
        labels_names: Dict[int, str]
            Dictionary mapping labels to their names.
    """
    if version == "corrected":
        data_file = os.path.join(data_path, "Indian_pines_corrected.mat")
    else:
        data_file = os.path.join(data_path, "Indian_pines.mat")
    data = loadmat(data_file)['indian_pines_corrected']
    labels = loadmat(os.path.join(data_path,
                                'Indian_pines_gt.mat'))['indian_pines_gt']
    labels_names = {
            0: 'Undefined',
            1: 'Alfalfa',
            2: 'Corn-notill',
            3: 'Corn-mintill',
            4: 'Corn',
            5: 'Grass-pasture',
            6: 'Grass-trees',
            7: 'Grass-pasture-mowed',
            8: 'Hay-windrowed',
            9: 'Oats',
            10: 'Soybean-notill',
            11: 'Soybean-mintill',
            12: 'Soybean-clean',
            13: 'Wheat',
            14: 'Woods',
            15: 'Buildings-Grass-Trees-Drives',
            16: 'Stone-Steel-Towers'
        }
    return data, labels, labels_names

def read_pavia_center(data_path: str) -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    """
    Read Pavia Center hyperspectral data.
    Parameters
        data_path: str
            Path to the data folder
    Returns
        data: ndarray
            Data array of shape (1096, 715, 102).
        labels: ndarray
            Labels array of shape (1096, 715).
        labels_names: Dict[int, str]
            Dictionary mapping labels to their names.
    """
    data_file = os.path.join(data_path, "Pavia.mat")
    data = loadmat(data_file)['pavia']
    labels = loadmat(os.path.join(data_path,
                                'Pavia_gt.mat'))['pavia_gt']
    labels_names = {
            0: 'Undefined',
            1: 'Asphalt',
            2: 'Meadows',
            3: 'Gravel',
            4: 'Trees',
            5: 'Painted metal sheets',
            6: 'Bare Soil',
            7: 'Bitumen',
            8: 'Self-Blocking Bricks',
            9: 'Shadows'
        }
    return data, labels, labels_names


def read_pavia_university(data_path: str) -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    """
    Read Pavia University hyperspectral data.

    Parameters
        data_path: str
            Path to the data folder

    Returns
    data: ndarray
        Data array of shape (610, 340, 103).
    labels: ndarray
        Labels array of shape (610, 340).
    labels_names: Dict[int, str]
        Dictionary mapping labels to their names.
    """
    data_file = os.path.join(data_path, "PaviaU.mat")
    data = loadmat(data_file)['paviaU']
    labels = loadmat(os.path.join(data_path,
                                'PaviaU_gt.mat'))['paviaU_gt']
    labels_names = {
            0: 'Undefined',
            1: 'Asphalt',
            2: 'Meadows',
            3: 'Gravel',
            4: 'Trees',
            5: 'Painted metal sheets',
            6: 'Bare Soil',
            7: 'Bitumen',
            8: 'Self-Blocking Bricks',
            9: 'Shadows'
        }
    return data, labels, labels_names


def read_kennedy_space_center(data_path: str) -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    """
    Read Kennedy Space Center hyperspectral data.

    Parameters
        data_path: str
            Path to the data folder

    Returns
        data: ndarray
            Data array of shape (512, 614, 176).
        labels: ndarray
            Labels array of shape (512, 614).
        labels_names: Dict[int, str]
            Dictionary mapping labels to their names.
    """

    data_file = os.path.join(data_path, "KSC.mat")
    data = loadmat(data_file)['KSC']
    labels = loadmat(os.path.join(data_path,
                                'KSC_gt.mat'))['KSC_gt']
    labels_names = {
            0: 'Undefined',
            1: 'Scrub',
            2: 'Willow swamp',
            3: 'Cabbage palm hammock',
            4: 'Cabbage palm/oak hammock',
            5: 'Slash pine',
            6: 'Oak/broadleaf hammock',
            7: 'Hardwood swamp',
            8: 'Graminoid marsh',
            9: 'Spartina marsh',
            10: 'Cattail marsh',
            11: 'Salt marsh',
            12: 'Mud flats',
            13: 'Water'
        }
    return data, labels, labels_names


def read_botswana(data_path: str) -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    """
    Read Botswana hyperspectral data.
    Parameters
        data_path: str
            Path to the data folder
    Returns
        data: ndarray
            Data array of shape (1476, 256, 145).
        labels: ndarray
            Labels array of shape (1476, 256).
        labels_names: Dict[int, str]
            Dictionary mapping labels to their names.
    """
    data_file = os.path.join(data_path, "Botswana.mat")
    data = loadmat(data_file)['Botswana']
    labels = loadmat(os.path.join(data_path,
                                'Botswana_gt.mat'))['Botswana_gt']
    labels_names = {
            0: 'Undefined',
            1: 'Water',
            2: 'Hippo grass',
            3: 'Floodplain grasses 1',
            4: 'Floodplain grasses 2',
            5: 'Reeds',
            6: 'Riparian',
            7: 'Firescar',
            8: 'Island interior',
            9: 'Acacia woodlands',
            10: 'Acacia shrublands',
            11: 'Acacia grasslands',
            12: 'Short mopane',
            13: 'Mixed mopane',
            14: 'Exposed soils'
        }
    return data, labels, labels_names


def read_data(dataset_path: str, data_name: str, small_dataset = False,
              reduce_factor=5) -> \
        Tuple[ndarray, ndarray, Dict[int, str]]:
    download_hyperspectral_data(dataset_path,data_name)
    if data_name.lower() == 'salinas':
        data, labels, labels_names = read_salinas(os.path.join(dataset_path, 'salinas'))
    elif data_name.lower() == 'indianpines':
        data, labels, labels_names = read_indian_pines(os.path.join(dataset_path, 'indianpines'))
    elif data_name.lower() in ['pavia', 'paviacenter', 'pavia_center']:
        data, labels, labels_names = read_pavia_center(os.path.join(dataset_path, 'Pavia'))
    elif data_name.lower() in ['paviau', 'paviauniversity', 'pavia_university']:
        data, labels, labels_names = read_pavia_university(os.path.join(dataset_path, 'PaviaU'))
    elif data_name.lower() in ['kennedy', 'kennedyspacecenter', 
                               'kennedy_space_center', 'ksc']:
        data, labels, labels_names = read_kennedy_space_center(os.path.join(dataset_path, 'KSC'))
    elif data_name.lower() == 'botswana':
        data, labels, labels_names = read_botswana(os.path.join(dataset_path, data_name))
    else:
        raise ValueError('Unknown dataset name: {}'.format(data_name))
    if small_dataset:
        data = data[::reduce_factor, ::reduce_factor]
        labels = labels[::reduce_factor,::reduce_factor]
    return data, labels, labels_names
