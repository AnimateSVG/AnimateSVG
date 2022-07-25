import pickle
import os
from os import listdir
from os.path import isfile, join
from xml.dom import minidom
from pathlib import Path
from matplotlib import image
from datetime import datetime
from shutil import copyfile
from skimage.metrics import mean_squared_error
from src.utils import logger
from src.data.svg_to_png import convert_svgs_in_folder


class Selector:
    """ Selector class for path relevance ordering. """

    def __init__(self, dir_svgs='./data/svgs', dir_path_selection='./data/path_selection',
                 dir_truncated_svgs='./data/truncated_svgs', dir_selected_paths='./data/selected_paths',
                 dir_decomposed_svgs='./data/decomposed_svgs'):
        """
        Args:
            dir_svgs (str): Directory containing SVGs to be sorted.
            dir_path_selection (str): Directory of logo folders containing PNGs of deleted paths.
            dir_truncated_svgs (str): Directory containing truncated SVGs to most relevant paths.
            dir_selected_paths (str): Directory containing decomposed SVGs selected by relevance ordering.
            dir_decomposed_svgs (str): Directory containing decomposed SVGs of all paths.

        """
        self.dir_svgs = dir_svgs
        self.dir_path_selection = dir_path_selection
        self.dir_truncated_svgs = dir_truncated_svgs
        self.dir_selected_paths = dir_selected_paths
        self.dir_decomposed_svgs = dir_decomposed_svgs

    @staticmethod
    def get_elements(doc):
        """ Retrieve all animation relevant elements from SVG.

        Args:
            doc (xml.dom.minidom.Document): XML minidom document from which to retrieve elements.

        Returns:
            list (xml.dom.minidom.Element): List of all elements in document

        """
        return doc.getElementsByTagName('path') + doc.getElementsByTagName('circle') + doc.getElementsByTagName(
            'ellipse') + doc.getElementsByTagName('line') + doc.getElementsByTagName(
            'polygon') + doc.getElementsByTagName('polyline') + doc.getElementsByTagName(
            'rect') + doc.getElementsByTagName('text')

    def delete_paths(self, logo):
        """ Function to iteratively delete single paths in an SVG and save remaining logo as PNG
        to Selector.dir_path_selection. Requires directory Selector.dir_decomposed_svgs.

        Args:
            logo (str): Name of logo (without file type ending).

        """
        Path(f'{self.dir_path_selection}/{logo}').mkdir(parents=True, exist_ok=True)
        doc = minidom.parse(f'{self.dir_svgs}/{logo}.svg')
        nb_original_elements = len(self.get_elements(doc))
        with open(f'{self.dir_path_selection}/{logo}/original.svg', 'wb') as file:
            file.write(doc.toprettyxml(encoding='iso-8859-1'))
        doc.unlink()
        for i in range(nb_original_elements):
            doc = minidom.parse(f'{self.dir_path_selection}/{logo}/original.svg')
            elements = self.get_elements(doc)
            path = elements[i]
            parent = path.parentNode
            parent.removeChild(path)
            with open(f'{self.dir_path_selection}/{logo}/without_id_{i}.svg', 'wb') as file:
                file.write(doc.toprettyxml(encoding='iso-8859-1'))
            doc.unlink()
        convert_svgs_in_folder(f'{self.dir_path_selection}/{logo}')

    def delete_paths_in_logos(self, logos):
        """ Iterate over list of logos to apply deletion of paths.

        Args:
            logos (list (str)): List of logos (without file type ending).

        """
        start = datetime.now()
        n_logos = len(logos)
        for i, logo in enumerate(logos):
            if i % 20 == 0:
                logger.info(f'Current logo {i+1}/{n_logos}: {logo}')
            self.delete_paths(logo)
        logger.info(f'Time: {datetime.now() - start}')

    @staticmethod
    def sort_by_relevance(path_selection_folder, excluded_paths, nr_paths_trunc=8):
        """ Sort paths in an SVG by relevance. Relevance of the path is measured by the MSE between the
        original logo and the logo resulting when deleting the path.
        The higher the MSE, the more relevant the given path.

        Args:
            path_selection_folder (str): Path to folder containing PNGs of the original logo and of the resulting logos
            when deleting each path.
            excluded_paths (list (int)): List of animation IDs that should not be considered as relevant. These paths
            will be assigned a relevance score of -1.
            nr_paths_trunc (int): Number of paths that should be kept as the most relevant ones.

        Returns:
            list (int), list(int), list (int), list (int): List of animation IDs sorted by relevance (descending),
            sorted list of MSE scores (descending), list of MSE scores of paths that were missed, list of animation IDs
            of paths that were misses due to exclusion.

        """
        nr_paths = len([name for name in os.listdir(path_selection_folder)
                        if os.path.isfile(os.path.join(path_selection_folder, name))]) - 1
        relevance_scores = []
        missed_scores, missed_paths = [], []
        img_origin = image.imread(os.path.join(path_selection_folder, "original.png"))
        logo = path_selection_folder.split('/')[-1]
        counter = 0
        for i in range(nr_paths):
            img_reduced = image.imread(os.path.join(path_selection_folder, "without_id_{}.png".format(i)))
            try:
                decomposed_id = f'{logo}_{i}'
                if decomposed_id in excluded_paths:
                    missed_mse = mean_squared_error(img_origin, img_reduced)
                    missed_scores.append(missed_mse)
                    missed_paths.append(decomposed_id)
                    logger.warning(f'No embedding for path {decomposed_id}, actual MSE would be: {missed_mse}')
                    mse = -1
                else:
                    mse = mean_squared_error(img_origin, img_reduced)
            except ValueError as e:
                logger.warning(f'Could not calculate MSE for path {logo}_{i} '
                               f'- Error message: {e}')
                counter += 1
                mse = -1
            relevance_scores.append(mse)
        relevance_score_ordering = list(range(nr_paths))
        relevance_score_ordering.sort(key=lambda x: relevance_scores[x], reverse=True)
        relevance_score_ordering = relevance_score_ordering[0:nr_paths_trunc]
        missed_relevant_scores, missed_relevant_paths = list(), list()
        for i in range(len(missed_scores)):
            score = missed_scores[i]
            if score >= relevance_scores[relevance_score_ordering[-1]]:
                missed_relevant_scores.append(score)
                missed_relevant_paths.append(missed_paths[i])
        if len(missed_relevant_scores) > 0:
            logger.warning(f'Number of missed relevant paths due to embedding: {len(missed_relevant_scores)}')
        if counter > 0:
            logger.warning(f'Could not calculate MSE for {counter}/{nr_paths} paths')
        relevance_score_ordering = [id_ for id_ in relevance_score_ordering if relevance_scores[id_] != -1]
        return relevance_score_ordering, relevance_scores, missed_relevant_scores, missed_relevant_paths

    def select_paths(self, svgs_folder, excluded_paths):
        """ Iterate over a directory of SVG files and select relevant paths. Selected paths and original
        SVGs will be saved to Selector.dir_selected_paths/logo. Requires directory Selector.dir_path_selection.

        Args:
            svgs_folder (str): Directory containing SVG files from which to select relevant paths.
            excluded_paths (list (int)): List of animation IDs that should not be considered as relevant. These paths
            will be assigned a relevance score of -1.

        Returns:
            list (int): List of missed paths.

        """
        Path(self.dir_selected_paths).mkdir(parents=True, exist_ok=True)
        logos = [f[:-4] for f in listdir(svgs_folder) if isfile(join(svgs_folder, f))]
        start = datetime.now()
        missed_scores, missed_paths = list(), list()
        for i, logo in enumerate(logos):
            if i % 20 == 0:
                logger.info(f'Current logo: {i}/{len(logos)}')
            sorted_ids, sorted_mses, missed_relevant_scores, missed_relevant_paths = \
                self.sort_by_relevance(f'{self.dir_path_selection}/{logo}', excluded_paths)
            missed_scores.append(len(missed_relevant_scores))
            missed_paths.extend(missed_relevant_paths)
            copyfile(f'{svgs_folder}/{logo}.svg', f'{self.dir_selected_paths}/{logo}_path_full.svg')
            for j, id_ in enumerate(sorted_ids):
                copyfile(f'{self.dir_decomposed_svgs}/{logo}_{id_}.svg',
                         f'{self.dir_selected_paths}/{logo}_path_{j}.svg')
        logger.info(f'Total number of missed paths: {sum(missed_scores)}')
        logger.info(f'Time: {datetime.now() - start}')
        return missed_paths

    def truncate_svgs(self, svgs_folder, logos=None, excluded_paths=list(), nr_paths_trunc=8):
        """ Truncate SVGs to most relevant paths and save them to Selector.dir_truncated_svgs. Requires directory
        Selector.dir_path_selection.

        Args:
            svgs_folder (str): Directory containing SVG files from which to select relevant paths.
            logos (list): List of logos to be truncated.
            excluded_paths (list (int)): List of animation IDs that should not be considered as relevant. These paths
            will be assigned a relevance score of -1.
            nr_paths_trunc (int): Number of paths that should be kept as the most relevant ones.

        """
        Path(self.dir_truncated_svgs).mkdir(parents=True, exist_ok=True)
        start = datetime.now()
        logos = [f[:-4] for f in listdir(svgs_folder) if isfile(join(svgs_folder, f))] if logos is None else logos
        for i, logo in enumerate(logos):
            if i % 20 == 0:
                logger.info(f'Current logo {i}/{len(logos)}: {logo}')
            sorted_ids, _, _, _ = self.sort_by_relevance(f'{self.dir_path_selection}/{logo}',
                                                         excluded_paths, nr_paths_trunc)
            doc = minidom.parse(f'{svgs_folder}/{logo}.svg')
            original_elements = self.get_elements(doc)
            nb_original_elements = len(original_elements)
            for j in range(nb_original_elements):
                if j not in sorted_ids:
                    path = original_elements[j]
                    parent = path.parentNode
                    parent.removeChild(path)
                with open(f'{self.dir_truncated_svgs}/{logo}.svg', 'wb') as file:
                    file.write(doc.toprettyxml(encoding='iso-8859-1'))
            doc.unlink()
        logger.info(f'Time: {datetime.now() - start}')


def get_path_relevance(logo, pkl_file='data/meta_data/path_relevance_order.pkl'):
    """ Get path relevance ordering from saved pickle file.

    Args:
        logo (str): Name of logo for which to obtain path ordering.

    Returns:
        list (int): List of animation IDs sorted by relevance.

    """
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)
    path_relevance_order = df[df['logo'] == logo]
    return path_relevance_order.iloc[0]['relevance_order']
