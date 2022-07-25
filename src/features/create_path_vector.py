from sklearn.decomposition import PCA
from src.features.get_style_attributes_folder import get_style_attributes_folder
from src.features.get_svg_size_pos import *
from src.preprocessing.create_svg_embedding import *
from src.data.get_svg_meta_data import *
import matplotlib.pyplot as plt
from PIL import ImageColor
from itertools import chain
import numpy as np
import pickle
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def create_path_vectors(svg_folder, emb_file_path=None, fitted_pca=None, new_dim=10, use_ppa=False,
                        style=True, size=True, position=True, nr_commands=True, nr_paths_svg=True,
                        avg_cols_svg=None, avg_diff=True, train=True):
    """ Create path-level model input vectors from a given folder of SVGs.

    Args:
        svg_folder (str): Path of folder containing all SVGs.
        emb_file_path (str): Path of path embedding file.
        fitted_pca (object): Fitted PCA model to apply on path embeddings.
        new_dim (float): Scalar that defines number of PCs to keep, or alternatively variance that should be explained
                         by the kept PCs.
        use_ppa (bool): If True, Principal Polynomial Analysis (PPA) is integrated in the dimension reduction step.
        style (bool): If True, style attributes of paths (colors, stroke, opacity) are integrated in path vectors.
        size (bool): If True, size of path is integrated in path vectors.
        position (bool): If True, relative position of path is integrated in path vectors.
        nr_commands (bool): If True, number of commands the path consists of is integrated in path vectors.
        nr_paths_svg (bool): If True, number of paths in the respective SVG is integrated in path vectors.
        avg_cols_svg (bool): If True, the average color of the SVG is integrated in path vectors.
        avg_diff (bool): If True, the difference between the path color and the average SVG color is integrated in path vectors.
        train (bool): If True, training data is considered, else test data.

    Returns:
        pd.DataFrame: Dataframe which contains path vectors that can be used as model input.

    """
    if avg_cols_svg is None:
        avg_cols_svg = ['fill_r', 'fill_g', 'fill_b',
                        'stroke_r', 'stroke_g', 'stroke_b']
    if emb_file_path:
        with open(emb_file_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = apply_embedding_model_to_svgs(data_folder="../../data/decomposed_svgs", save=False)

    # use manually splitting after inspecting the logos (ratio should be around 80/20)
    logos_train = ['logo_{}'.format(i) for i in chain(range(147), range(192, 395))]  # 350
    logos_test = ['logo_{}'.format(i) for i in chain(range(147, 192), range(395, 437))]  # 87

    if train:
        df = df.loc[df['filename'].isin(logos_train)]
    else:
        df = df.loc[df['filename'].isin(logos_test)]

    if new_dim:
        df_meta = df.iloc[:, :2].reset_index(drop=True)
        df_emb = df.iloc[:, 2:]
        df_emb_red, fitted_pca = reduce_dim(df_emb, fitted_pca=fitted_pca, new_dim=new_dim, use_ppa=use_ppa)
        df = pd.concat([df_meta, df_emb_red.reset_index(drop=True)], axis=1)

    if style:
        st = _get_transform_style_elements(svg_folder)
        df = df.merge(st, how='left', on=['filename', 'animation_id'])
        if avg_cols_svg:
            df = _get_svg_avg(df, avg_cols_svg, avg_diff)

    if size:
        df['rel_width'] = df.apply(lambda row: get_relative_path_size(f"{svg_folder}/{row['filename']}.svg", row['animation_id'])[0], axis=1)
        df['rel_height'] = df.apply(lambda row: get_relative_path_size(f"{svg_folder}/{row['filename']}.svg", row['animation_id'])[1], axis=1)

    if position:
        df['rel_x_position'] = df.apply(lambda row: get_relative_path_pos(f"{svg_folder}/{row['filename']}.svg", row['animation_id'])[0], axis=1)
        df['rel_y_position'] = df.apply(lambda row: get_relative_path_pos(f"{svg_folder}/{row['filename']}.svg", row['animation_id'])[1], axis=1)

    if nr_paths_svg:
        meta_df = get_svg_meta_data(data_folder=svg_folder)
        df = df.merge(meta_df[['id', 'nb_groups']], how='left', left_on=['filename'], right_on=['id'])
        df.drop(['id'], axis=1, inplace=True)
        df = df.rename(columns={'nb_groups': 'nr_paths_svg'})

    if nr_commands:
        meta_df = get_svg_meta_data(data_folder=svg_folder)
        df['nr_commands'] = df.apply(lambda row: meta_df[meta_df['id'] == row['filename']].reset_index()['len_groups'][0][int(row['animation_id'])], axis=1)

    if train:
        return df, fitted_pca
    else:
        return df


def reduce_dim(data: pd.DataFrame, fitted_pca=None, new_dim=None, use_ppa=False, ppa_threshold=8):
    """ Reduces dimensionality of path embeddings.

    Args:
        data (pd.Dataframe): Path embedding data.
        fitted_pca (object): Fitted PCA model to apply on path embeddings.
        new_dim (float): Scalar that defines number of PCs to keep, or alternatively variance that should be explained
                         by the kept PCs.
        use_ppa: If True, Principal Polynomial Analysis (PPA) is integrated in the dimension reduction step.
        ppa_threshold (int): PPA threshold.

    Returns:
        pd.DataFrame, object: Dimension-reduced path embeddings, fitted PCA model

    """
    # 1. PPA #1
    # PCA to get Top Components

    def _ppa(data, N, D):
        pca = PCA(n_components=N)
        data = data - np.mean(data)
        _ = pca.fit_transform(data)
        U = pca.components_

        z = []

        # Removing Projections on Top Components
        for v in data:
            for u in U[0:D]:
                v = v - np.dot(u.transpose(), v) * u
            z.append(v)
        return np.asarray(z)

    X = np.array(data)

    if use_ppa:
        X = _ppa(X, X.shape[1], ppa_threshold)

    # 2. PCA
    # PCA Dim Reduction
    X = X - np.mean(X)
    if not fitted_pca:
        fitted_pca = PCA(n_components=new_dim, random_state=42)
        X = fitted_pca.fit_transform(X)
    else:
        X = fitted_pca.transform(X)

    # 3. PPA #2
    if use_ppa:
        X = _ppa(X, new_dim, ppa_threshold)

    emb_df = pd.DataFrame(X)
    emb_df.columns = ['emb_{}'.format(i) for i in range(emb_df.shape[1])]
    return emb_df, fitted_pca


def _get_transform_style_elements(svg_folder):
    # get local and global style elements and combine
    st = get_style_attributes_folder(svg_folder)
    st.dropna(inplace=True)
    # transform 'fill' hexacode into RGB channels
    for i, c in enumerate(['r', 'g', 'b']):
        st['fill_{}'.format(c)] = st.apply(lambda row: ImageColor.getcolor(row['fill'], 'RGB')[i], axis=1)

    # transform 'stroke' hexacode into RGB channels
    for i, c in enumerate(['r', 'g', 'b']):
        st['stroke_{}'.format(c)] = st.apply(lambda row: ImageColor.getcolor(row['stroke'], 'RGB')[i], axis=1)

    st.drop(['class_', 'href', 'fill', 'stroke'], inplace=True, axis=1)

    return st


def _get_svg_avg(df, columns, diff=True):
    for col in columns:
        df[f'svg_{col}'] = df.groupby('filename')[col].transform('mean')
        if diff:
            df[f'diff_{col}'] = df[col] - df[f'svg_{col}']
    return df


if __name__ == '__main__':
    path_embedding_pkl = "../../data/embeddings/path_embedding.pkl"
    train_df, fitted_pca = create_path_vectors("../../data/initial_svgs",
                                               emb_file_path=path_embedding_pkl,
                                               new_dim=10,  # explained variance (0.99) or nr. of components (10)
                                               nr_commands=False,  # "list index out of range"
                                               train=True)

    # Check number of principal components and plot of cumulative explained variance
    explained_variance = fitted_pca.explained_variance_ratio_
    print(f"Number of principal components = {len(explained_variance)}")
    print(f"Fitted pca: {fitted_pca}")
    #plt.plot(np.cumsum(explained_variance))
    #plt.xlabel('number of components')
    #plt.ylabel('cumulative explained variance')

    # save PCA
    pickle.dump(fitted_pca, open("../../models/pca_path_embedding.sav", 'wb'))
    print(f"Fitted PCA saved.")

    train_df.to_csv('../../data/path_selector/path_selector_train.csv', index=False)
    print('Train data created and saved.')

    test_df = create_path_vectors("../../data/initial_svgs",
                                  emb_file_path=path_embedding_pkl,
                                  fitted_pca=fitted_pca,
                                  nr_commands=False,  # "list index out of range"
                                  train=False)
    test_df.to_csv('../../data/path_selector/path_selector_test.csv', index=False)
    print('Test data created and saved.')
