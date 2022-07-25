import os
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.animations.get_path_probabilities import get_path_probabilities
from src.animations.insert_animation import create_animated_svg
from src.preprocessing.sort_paths import get_path_relevance

# Specify path to pkl file containing path labels
animation_path_label = "data/path_selector/path_selector_label.pkl"

# Specify path to pkl file containing path relevance order
path_relevance_order = "data/meta_data/path_relevance_order.pkl"


def create_random_animations(folder, nb_animations, split_df=True, very_random=False):
    """ Create random animations. Animation vectors are saved in data/animated_svgs_dataframes.

    Args:
        folder (str): Path of folder containing all SVGs.
        nb_animations (int): Number of random animations per SVG.
        split_df (bool): If true, animation vectors are saved to multiple dataframes (one dataframe per SVG).
                            If false, animation vectors are saved to one dataframe and returned.
        very_random (bool): If true, random seed is shuffled.

    Returns:
        pd.DataFrame: If split_df=False, dataframe containing all animation vectors is returned.

    """
    Path("data/animated_svgs_dataframes").mkdir(parents=True, exist_ok=True)
    if split_df:
        create_multiple_df(folder, nb_animations, very_random)
    else:
        return create_one_df(folder, nb_animations, very_random)


def create_multiple_df(folder, nb_animations, very_random):
    """ Create random animations. Animation vectors are saved to one dataframe per SVG.

    Args:
        folder (str): Path of folder containing all SVGs.
        nb_animations (int): Number of random animations per SVG.
        very_random (bool): If true, random seed is shuffled.

    """
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            df = pd.DataFrame.from_records(_create_multiple_df(folder, file, nb_animations, very_random))
            output = open(f"data/animated_svgs_dataframes/{file.replace('.svg', '')}_animation_vectors.pkl", 'wb')
            pickle.dump(df, output)
            output.close()


def _create_multiple_df(folder, file, nb_animations, very_random):
    relevant_animation_ids = get_path_relevance(file.replace('.svg', ''), pkl_file=path_relevance_order)
    path_probs = get_path_probabilities(file.replace('.svg', ''), relevant_animation_ids, pkl_file=animation_path_label)
    file = folder + "/" + file
    for random_seed in range(nb_animations):
        if very_random:
            random_seed = random.randint(0, 1000)
        animation_vectors, animated_order_ids, backend_mapping = random_animation_vector(
            nr_animations=len(relevant_animation_ids),
            path_probs=path_probs,
            seed=random_seed)
        # Create list of animation IDs that were animated
        animated_animation_ids = []
        for i in range(len(animated_order_ids)):
            animated_animation_ids.append(relevant_animation_ids[animated_order_ids[i]])
        begin_values, _ = create_animated_svg(file, animated_animation_ids, animation_vectors, str(random_seed))
        for j in range(len(animated_animation_ids)):
            yield dict(file=f'{file.split("/")[-1].replace(".svg", "")}_animation_{random_seed}',
                       animation_id=animated_animation_ids[j],
                       order_id=animated_order_ids[j],
                       path_prob=path_probs[j],
                       begin_value=begin_values[j],
                       model_output=animation_vectors[j],
                       animated_animation_ids=animated_animation_ids,
                       animated_order_ids=animated_order_ids,
                       backend_mapping=backend_mapping)


def create_one_df(folder, nb_animations, very_random):
    """ Create random animations. Animation vectors are saved to one dataframe.

    Args:
        folder (str): Path of folder containing all SVGs.
        nb_animations (int): Number of random animations per SVG.
        very_random (bool): If true, random seed is shuffled.

    Returns:
        pd.DataFrame: Dataframe containing all animation vectors.

    """
    df = pd.DataFrame.from_records(_create_one_df(folder, nb_animations, very_random))

    date_time = datetime.now().strftime('%H%M')
    output = open(f"data/animated_svgs_dataframes/{date_time}_animation_vectors.pkl", 'wb')
    pickle.dump(df, output)
    output.close()

    return df


def _create_one_df(folder, nb_animations, very_random):
    for file in os.listdir(folder):
        if file.endswith(".svg"):
            relevant_animation_ids = get_path_relevance(file.replace('.svg', ''), pkl_file=path_relevance_order)
            path_probs = get_path_probabilities(file.replace('.svg', ''), relevant_animation_ids,
                                                pkl_file=animation_path_label)
            file = folder + "/" + file
            for random_seed in range(nb_animations):
                if very_random:
                    random_seed = random.randint(0, 1000)
                animation_vectors, animated_order_ids, backend_mapping = random_animation_vector(
                    nr_animations=len(relevant_animation_ids),
                    path_probs=path_probs,
                    seed=random_seed)
                # Create list of animation IDs that were animated
                animated_animation_ids = []
                for i in range(len(animated_order_ids)):
                    animated_animation_ids.append(relevant_animation_ids[animated_order_ids[i]])
                begin_values, _ = create_animated_svg(file, animated_animation_ids, animation_vectors, str(random_seed))
                for j in range(len(animated_animation_ids)):
                    yield dict(file=f'{file.split("/")[-1].replace(".svg", "")}_animation_{random_seed}',
                               animation_id=animated_animation_ids[j],
                               order_id=animated_order_ids[j],
                               path_prob=path_probs[j],
                               begin_value=begin_values[j],
                               model_output=animation_vectors[j],
                               animated_animation_ids=animated_animation_ids,
                               animated_order_ids=animated_order_ids,
                               backend_mapping=backend_mapping)


def random_animation_vector(nr_animations, path_probs=None, animation_type_prob=None, seed=73):
    """ Generate random animation vectors.

    Format of animation vectors: [translate, scale, rotate, skew, fill, opacity, translate_from_1, translate_from_2, scale_from, rotate_from, skew_from_1, skew_from_2].

    Note: nr_animations must match length of path_probs.

    Args:
        nr_animations (int): Number of animation vectors that are generated.
        path_probs (list): Specifies how likely it is that a path gets animated.
        animation_type_prob (list): Specifies probabilities of animation types. Default is a uniform distribution.
        seed (int): Random seed.

    Returns:
        ndarray, list: Array of 12 dimensional random animation vectors, list of IDs of elements that were animated.

    """
    if path_probs is None:
        path_probs = [1 / 2] * nr_animations
    if animation_type_prob is None:
        animation_type_prob = [1 / 6] * 6

    random.seed(seed)
    np.random.seed(seed)
    vec_list = []
    animated_order_ids = []
    backend_mapping = []
    for i in range(nr_animations):
        animate = np.random.choice(a=[False, True], p=[1 - path_probs[i], path_probs[i]])
        if not animate:
            # vec_list.append(np.array([int(0)] * 6 + [float(-1.0)] * 6, dtype=object))
            backend_mapping.append(0)
        else:
            vec = np.array([int(0)] * 6 + [float(-1.0)] * 6, dtype=object)
            animation_type = np.random.choice(a=[0, 1, 2, 3, 4, 5], p=animation_type_prob)
            vec[animation_type] = 1
            if animation_type == 0:  # translate
                vec[6] = random.uniform(0, 1)
                vec[7] = random.uniform(0, 1)
            if animation_type == 1:  # scale
                vec[8] = random.uniform(0, 1)
            if animation_type == 2:  # rotate
                vec[9] = random.uniform(0, 1)
            if animation_type == 3:  # skew
                vec[10] = random.uniform(0, 1)
                vec[11] = random.uniform(0, 1)
            vec_list.append(vec)
            animated_order_ids.append(i)
            backend_mapping.append(1)
    return np.array(vec_list), animated_order_ids, backend_mapping


def combine_dataframes(folder):
    """ Function to combine all dataframes saved as pkl in a given folder.

    Args:
        folder (str): Path of folder containing dataframes.

    Returns:
        pd.DataFrame: Concatenated dataframe.

    """
    df_list = []
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            with open(f'{folder}/{file}', 'rb') as f:
                df_list.append(pickle.load(f))
    return pd.concat(df_list).reset_index(drop=True)


def create_backend_mapping_df(df):
    """ Function to create backend mapping for labeling website.

    Args:
        df (pd.DataFrame): Unprocessed dataframe containing backend mapping.

    Returns:
        pd.DataFrame: Processed dataframe containing backend mapping.

    """
    df = df.set_index('file')
    df = df['backend_mapping'].apply(pd.Series)
    df = df[~df.index.duplicated(keep='first')]
    df = df.reset_index(col_level=1)
    return df
