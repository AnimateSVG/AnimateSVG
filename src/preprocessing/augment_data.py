import os
import torch
import pickle
import pandas as pd
import numpy as np
import random
from datetime import datetime
from xml.dom import minidom

from src.features.create_path_vector import reduce_dim
from src.preprocessing.configs.deepsvg.hierarchical_ordered import Config
from src.preprocessing.deepsvg.svglib.geom import Point
from src.preprocessing.deepsvg.svglib.svg import SVG
from src.preprocessing.deepsvg.difflib.tensor import SVGTensor
from src.preprocessing.deepsvg.utils.utils import batchify
from src.preprocessing.deepsvg import utils
from src.features.get_svg_size_pos import get_svg_bbox, get_relative_path_pos, get_midpoint_of_path_bbox


def augment_data(folder='data/svgs',
                 nb_augmentations=2,
                 df_dir='data/path_selector/path_selector_train.csv',
                 embedding_model='models/deepSVG_hierarchical_ordered.pth.tar',
                 pca_model='models/pca_path_embedding.sav',
                 seed=None,
                 save=True):
    """ Data augmentation by randomly scaling and translating each path in an SVG.

    Scaling: Path is scaled by a random factor in the interval [0.8, 1.2].

    Translation: Path is translated by a random translation vector t where tx and ty are sampled independently in the
    interval [-10, 10].

    Args:
        folder (str): Path of folder containing all SVGs.
        nb_augmentations (int): Number of augmentations per path.
        df_dir (str): Directory of data that is augmented.
        embedding_model (str): Path of embedding model.
        pca_model (str): Path of PCA model.
        seed (int): Random seed.
        save (bool): If true, resulting dataframe is saved.

    Returns:
        pd.DataFrame: Dataframe containing metadata of SVGs.

    """
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)

    df_original = pd.read_csv(df_dir)
    df_original.drop(['emb_0', 'emb_1', 'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7', 'emb_8', 'emb_9'],
                     axis=1, inplace=True)

    df_embed_aug = pd.DataFrame.from_records(
        _get_embedding_of_augmented_data(folder, embedding_model, nb_augmentations, seed))

    # Drop rows where embedding contains nan values
    df_embed_aug['temp'] = df_embed_aug['embedding'].apply(lambda row: np.isnan(row.numpy()).any())
    df_embed_aug = df_embed_aug[~df_embed_aug['temp']]

    # Apply PCA to embedding
    df_emb = df_embed_aug['embedding'].apply(lambda row: row.numpy()[0][0][0]).apply(pd.Series)
    fitted_pca = pickle.load(open(pca_model, 'rb'))
    df_emb_red, _ = reduce_dim(df_emb, fitted_pca=fitted_pca)

    # Concatenate dataframes and drop unnecessary columns
    df = pd.concat([df_embed_aug.reset_index(drop=True), df_emb_red.reset_index(drop=True)], axis=1)
    df.drop(['temp', 'embedding'], axis=1, inplace=True)

    df_full = df.merge(df_original, how='left', on=['filename', 'animation_id'])
    df_full.dropna(inplace=True)

    df_full['rel_width'] = df_full.apply(lambda row: row['scale'] * row['rel_width'], axis=1)
    df_full['rel_height'] = df_full.apply(lambda row: row['scale'] * row['rel_height'], axis=1)

    df_full['translation_factor'] = df_full.apply(
        lambda row: _get_translation_factor(f"{folder}/{row['filename']}.svg", row['animation_id'], row['translate_x'], row['translate_y']),
        axis=1)
    df_full['translation_factor_x'] = df_full['translation_factor'].apply(lambda row: row[0])
    df_full['translation_factor_y'] = df_full['translation_factor'].apply(lambda row: row[1])

    df_full['rel_x_position'] = df_full.apply(lambda row: row['translation_factor_x'] * row['rel_x_position'], axis=1)
    df_full['rel_y_position'] = df_full.apply(lambda row: row['translation_factor_y'] * row['rel_y_position'], axis=1)

    df_full.drop(['translate_x', 'translate_y', 'scale', 'translation_factor', 'translation_factor_x', 'translation_factor_y'], axis=1)

    # Introduce noise to animation vectors for model stability
    #if introduce_noise_to_animation_vectors:
    #    for i in range(6, 12):
    #        df_full[f'an_vec_{i}'] = df_full[f'an_vec_{i}'].apply(
    #            lambda row: row + (0.2 * random.random() - 0.1) if 0.1 < row < 0.9 else row)

    col_order = ['filename', 'animation_id', 'nb_augmentation'] \
                + [f'emb_{i}' for i in range(10)] \
                + ['fill_r', 'fill_g', 'fill_b', 'svg_fill_r', 'svg_fill_g', 'svg_fill_b',
                   'diff_fill_r', 'diff_fill_g', 'diff_fill_b', 'rel_height', 'rel_width',
                   'rel_x_position', 'rel_y_position', 'nr_paths_svg']

    df_full = df_full[col_order]

    if save:
        date_time = datetime.now().strftime('%H%M')
        df_full.to_csv(f'data/path_selector/model_1_train_nb_augmentations_{nb_augmentations}_{date_time}.csv', index=False)

    return df_full


def _get_embedding_of_augmented_data(folder, embedding_model, nb_augmentations, seed=None):
    """ Get embedding and augmentation parameters. """
    # Load pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = cfg.make_model().to(device)
    utils.load_model(embedding_model, model)
    model.eval();

    for file in os.listdir(folder):
        if file.endswith(".svg"):
            svg_decomposed_docs = _decompose_svg(f'{folder}/{file}')
            for i, doc in enumerate(svg_decomposed_docs):
                svg = SVG.from_str(doc)
                svg = _simplify(svg, normalize=True)

                for j in range(nb_augmentations):
                    # The following parameters are defined in the deepSVG config:
                    model_args = ['commands', 'args', 'commands', 'args']

                    # The following parameters are defined in class SVGDataset:
                    MAX_NUM_GROUPS = 8
                    MAX_SEQ_LEN = 30
                    MAX_TOTAL_LEN = 50
                    PAD_VAL = -1

                    svg_aug, dx, dy, factor = _preprocess(svg, seed=seed)

                    t_sep, fillings = svg_aug.to_tensor(concat_groups=False, PAD_VAL=PAD_VAL), svg_aug.to_fillings()
                    # Note: DeepSVG can only handle 8 paths in a SVG and 30 sequences per path
                    if len(t_sep) > 8:
                        # print(f"SVG has more than 30 segments.")
                        t_sep = t_sep[0:8]
                        fillings = fillings[0:8]

                    for k in range(len(t_sep)):
                        if len(t_sep[k]) > 30:
                            # print(f"Path nr {i} has more than 30 segments.")
                            t_sep[k] = t_sep[k][0:30]

                    res = {}
                    pad_len = max(MAX_NUM_GROUPS - len(t_sep), 0)

                    t_sep.extend([torch.empty(0, 14)] * pad_len)
                    fillings.extend([0] * pad_len)

                    t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=PAD_VAL).add_eos().add_sos().pad(
                        seq_len=MAX_TOTAL_LEN + 2)]

                    t_sep = [SVGTensor.from_data(t, PAD_VAL=PAD_VAL, filling=f).add_eos().add_sos().pad(
                        seq_len=MAX_SEQ_LEN + 2) for t, f in zip(t_sep, fillings)]

                    for arg in set(model_args):
                        if "_grouped" in arg:
                            arg_ = arg.split("_grouped")[0]
                            t_list = t_grouped
                        else:
                            arg_ = arg
                            t_list = t_sep

                        if arg_ == "tensor":
                            res[arg] = t_list

                        if arg_ == "commands":
                            res[arg] = torch.stack([t.cmds() for t in t_list])

                        if arg_ == "args_rel":
                            res[arg] = torch.stack([t.get_relative_args() for t in t_list])
                        if arg_ == "args":
                            res[arg] = torch.stack([t.args() for t in t_list])

                    model_args = batchify((res[key] for key in model_args), device)

                    with torch.no_grad():
                        z = model(*model_args, encode_mode=True)

                    yield dict(filename=file.split('.svg')[0], animation_id=i, nb_augmentation=j,
                               translate_x=dx, translate_y=dy, scale=factor, embedding=z)


def _decompose_svg(file):
    """ Decompose an SVG into its paths. """
    svg_doc = minidom.parse(file)
    elements = _store_svg_elements(svg_doc)
    num_elements = len(elements)

    decomposed_docs = []
    for i in range(num_elements):
        # load SVG again: necessary because we delete elements in each loop
        doc_temp = minidom.parse(file)
        elements_temp = _store_svg_elements(doc_temp)
        # select all elements besides one
        elements_temp_remove = elements_temp[:i] + elements_temp[i + 1:]
        for element in elements_temp_remove:
            # Check if current element is referenced clip path
            if not element.parentNode.nodeName == "clipPath":
                parent = element.parentNode
                parent.removeChild(element)
        decomposed_docs.append(doc_temp.toxml())
        doc_temp.unlink()

    return decomposed_docs


def _store_svg_elements(svg_doc):
    return svg_doc.getElementsByTagName('path') + svg_doc.getElementsByTagName('circle') + \
           svg_doc.getElementsByTagName('ellipse') + svg_doc.getElementsByTagName('line') + \
           svg_doc.getElementsByTagName('polygon') + svg_doc.getElementsByTagName('polyline') + \
           svg_doc.getElementsByTagName('rect') + svg_doc.getElementsByTagName('text')


def _simplify(svg, normalize=True):
    svg = svg.canonicalize(normalize=normalize)
    svg = svg.simplify_heuristic()
    return svg.normalize()


def _preprocess(svg, augment=True, seed=None):
    dx = dy = factor = 0
    if augment:
        svg, dx, dy, factor = _augment(svg, seed)
    return svg.numericalize(256), dx, dy, factor


def _augment(svg, seed=None):
    """ Scale and translate randomly. """
    if seed is not None:
        random.seed(seed)
    dx, dy = (-10 + 20 * random.random(), -10 + 20 * random.random())
    factor = 0.8 + 0.4 * random.random()
    return svg.zoom(factor).translate(Point(dx, dy)), dx, dy, factor


def _get_translation_factor(file, animation_id, dx, dy):
    """ Function to get relative position of a path in an SVG file."""
    try:
        initial_x, initial_y = get_relative_path_pos(file, animation_id)

        path_midpoint_x, path_midpoint_y = get_midpoint_of_path_bbox(file, animation_id)
        path_midpoint_x = path_midpoint_x + dx
        path_midpoint_y = path_midpoint_y + dy
        svg_xmin, svg_xmax, svg_ymin, svg_ymax = get_svg_bbox(file)
        new_x = (path_midpoint_x - svg_xmin) / (svg_xmax - svg_xmin)
        new_y = (path_midpoint_y - svg_ymin) / (svg_ymax - svg_ymin)

        factor_x = new_x / initial_x
        factor_y = new_y / initial_y
    except Exception as e:
        print(f'File {file}, animation ID {animation_id}: Translation factor cannot be computed and is set to 0.01. {e}.')
        factor_x = 0.01
        factor_y = 0.01

    return factor_x, factor_y


if __name__ == '__main__':
    os.chdir('../..')
    df = augment_data(folder='data/augment_svgs',
                      nb_augmentations=2,
                      save=True)
