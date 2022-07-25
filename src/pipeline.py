from xml.dom import minidom

import pickle5
from PIL import ImageColor
from svgpathtools import svg2paths
from src.animations import *
from src.features import get_style_attributes_path
from src.features.create_path_vector import reduce_dim
from src.features.get_svg_size_pos import get_svg_size, get_svg_bbox, get_relative_path_pos, get_relative_path_size, \
    get_begin_values_by_starting_pos
from src.models import config
from src.models.entmoot_functions import *
from src.models.train_animation_predictor import *
from src.preprocessing.configs.deepsvg.hierarchical_ordered import Config
from src.preprocessing.deepsvg import utils
from src.preprocessing.deepsvg.difflib.tensor import SVGTensor
from src.preprocessing.deepsvg.svglib.svg import SVG
from src.preprocessing.deepsvg.utils.utils import batchify


class Logo:
    """ Logo class used for pipeline on website. """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path of logo.

        """
        self.data_dir = data_dir
        self.parsed_doc = minidom.parse(data_dir)
        self.nr_paths = len(self._store_svg_elements(self.parsed_doc))
        self.animation_ids = [*range(self.nr_paths)]
        self.width, self.height = get_svg_size(data_dir)
        self.xmin_svg, self.xmax_svg, self.ymin_svg, self.ymax_svg = get_svg_bbox(data_dir)

    def print_logo_information(self):
        """ Print information of given logo, such as data directory, parsed doc, number of elements, animation IDs,
        width, height and coordinates of bounding box. """
        print('--------------------------- Logo Information ---------------------------')
        print(f'data_dir: {self.data_dir}')
        print(f'parsed_doc: {self.parsed_doc}')
        print(f'nr_paths: {self.nr_paths}')
        print(f'animation_ids: {self.animation_ids}')
        print(f'width, height: {self.width}, {self.height}')
        print(f'bbox: {self.xmin_svg}, {self.xmax_svg}, {self.ymin_svg}, {self.ymax_svg}')

    def animate(self, model='all'):
        """ Automatically animate a logo and save animations in the same folder directory.

        Args:
              model (str): Chosen model for generation of animations. Should be 'opt' for entmoot optimization,
                            'backprop' for generation model using aesthetic evaluation loss function,
                            or 'all' if both models should be applied to generate animations.

        """
        if 'preprocessed' not in self.data_dir:
            self.preprocess()

        # Create input for model 1
        df = self.create_df(pca_model=config.pca_path)

        # Retrieve model 1 prediction and extract relative position to midpoint of animated paths of SVG
        df = retrieve_m1_predictions(df)
        df = retrieve_animation_midpoints(df, data_dir=os.path.dirname(self.data_dir), drop=True)

        # Scale features
        scaler = pickle5.load(open(config.scaler_path, 'rb'))
        df[config.sm_features] = scaler.transform(df[config.sm_features])

        svg_animations = pd.DataFrame({'filename': [], 'animation_id': [], 'animation_vector': [], 'model': []})

        if (model == 'opt') | (model == 'all'):
            entmoot_animations = Logo._create_animation_entmoot(df)
            svg_animations = svg_animations.append(entmoot_animations)
            svg_animations = svg_animations.fillna('entmoot')

        if (model == 'backprop') | (model == 'all'):
            loss_animations = Logo._create_animation_loss(df)
            svg_animations = svg_animations.append(loss_animations)
            svg_animations = svg_animations.fillna('backprop')

        for i, row in svg_animations.iterrows():
            try:
                self._insert_animation(row['animation_id'], row['animation_vector'], filename_suffix=row['model'])
            except FileNotFoundError:
                print(f"File not found: {row['filename']}")
                pass

    @staticmethod
    def _create_animation_entmoot(df):
        # Extract path vectors as list
        path_vectors = df[config.sm_features].values.tolist()

        # Load ENTMOOT optimizer to data
        with open("models/entmoot_optimizer_final.pkl", "rb") as f:
            optimizer = pickle5.load(f)

        # Load surrogate model for function evaluations
        func = SurrogateModelFNN()

        # Predict and store animation vectors
        an_vec_preds = []
        for i in range(len(path_vectors)):
            opt_x, opt_y, _ = entmoot_predict(optimizer, func, path_vectors[i], n_calls=5)
            an_vec_preds.append(opt_x)

        df['animation_vector'] = an_vec_preds

        gb = [df.groupby('filename')[column].apply(list) for column in
              'animation_id animation_vector'.split()]
        svg_animations = pd.concat(gb, axis=1).reset_index()

        return svg_animations

    @staticmethod
    def _create_animation_loss(df):
        # Extract path vectors from dataframe
        path_vectors = torch.tensor(df[config.sm_features].to_numpy(), dtype=torch.float)

        # Create instance of animation prediction model
        ap = AnimationPredictor()
        ap.load_state_dict(torch.load(config.ap_state_dict_path))

        an_predictions = ap(path_vectors)
        an_predictions = create_animation_vector(an_predictions)

        df['animation_vector'] = an_predictions.detach().numpy().tolist()

        gb = [df.groupby('filename')[column].apply(list) for column in
              'animation_id animation_vector'.split()]
        svg_animations = pd.concat(gb, axis=1).reset_index()

        return svg_animations

    def preprocess(self, percent=50):
        """ Add attribute "animation_id" to all elements in an SVG and expands/inserts a viewbox.

         Args:
             percent (int): Percentage in %: How much do we want to expand the viewbox? Default is 50%.

        """
        if 'preprocessed' not in self.data_dir:
            # Expand/insert viewbox
            x, y = '', ''
            # get width and height of logo
            try:
                width = self.parsed_doc.getElementsByTagName('svg')[0].getAttribute('width')
                height = self.parsed_doc.getElementsByTagName('svg')[0].getAttribute('height')
                if not width[-1].isdigit():
                    width = width.replace('px', '').replace('pt', '')
                if not height[-1].isdigit():
                    height = height.replace('px', '').replace('pt', '')
                x = float(width)
                y = float(height)
                check = True
            except:
                check = False
            if not check:
                # get bounding box of svg
                xmin_svg, xmax_svg, ymin_svg, ymax_svg = 0, 0, 0, 0
                paths, attributes = svg2paths(self.data_dir)
                for path in paths:
                    xmin, xmax, ymin, ymax = path.bbox()
                    if xmin < xmin_svg:
                        xmin_svg = xmin
                    if xmax > xmax_svg:
                        xmax_svg = xmax
                    if ymin < ymin_svg:
                        ymin_svg = ymin
                    if ymax > ymax_svg:
                        ymax_svg = ymax
                    x = xmax_svg - xmin_svg
                    y = ymax_svg - ymin_svg

            # Check if viewBox exists
            if self.parsed_doc.getElementsByTagName('svg')[0].getAttribute('viewBox') == '':
                v1, v2, v3, v4 = 0, 0, 0, 0
                # Calculate new viewBox values
                x_new = x * (100 + percent) / 100
                y_new = y * (100 + percent) / 100
            else:
                v1 = float(
                    self.parsed_doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[0].replace('px',
                                                                                                                 '').replace(
                        'pt', '').replace(',', ''))
                v2 = float(
                    self.parsed_doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[1].replace('px',
                                                                                                                 '').replace(
                        'pt', '').replace(',', ''))
                v3 = float(
                    self.parsed_doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[2].replace('px',
                                                                                                                 '').replace(
                        'pt', '').replace(',', ''))
                v4 = float(
                    self.parsed_doc.getElementsByTagName('svg')[0].getAttribute('viewBox').split(' ')[3].replace('px',
                                                                                                                 '').replace(
                        'pt', '').replace(',', ''))
                x = v3
                y = v4
                # Calculate new viewBox values
                x_new = x * percent / 100
                y_new = y * percent / 100
            x_translate = - x * percent / 200
            y_translate = - y * percent / 200
            coordinates = str(v1 + x_translate) + ' ' + str(v2 + y_translate) + ' ' + str(v3 + x_new) + ' ' + str(
                v4 + y_new)
            self.parsed_doc.getElementsByTagName('svg')[0].setAttribute('viewBox', coordinates)

            # Insert animation_id
            elements = self._store_svg_elements(self.parsed_doc)
            for i in range(len(elements)):
                elements[i].setAttribute('animation_id', str(i))

            # create new file and update data_dir
            textfile = open(f"{self.data_dir.replace('.svg', '')}_preprocessed.svg", 'wb')
            textfile.write(self.parsed_doc.toprettyxml(encoding="iso-8859-1"))
            textfile.close()
            self.data_dir = f"{self.data_dir.replace('.svg', '')}_preprocessed.svg"

    def decompose_svg(self):
        """ Decompose a SVG into its paths.

        Returns:
            list(xml.dom.minidom.Document): Decomposed SVG as list of paths.

        """
        elements = Logo._store_svg_elements(self.parsed_doc)
        num_elements = len(elements)

        decomposed_docs = []
        for i in range(num_elements):
            # load SVG again: necessary because we delete elements in each loop
            doc_temp = minidom.parse(self.data_dir)
            elements_temp = Logo._store_svg_elements(doc_temp)
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

    @staticmethod
    def _store_svg_elements(parsed_doc):
        return parsed_doc.getElementsByTagName('path') + parsed_doc.getElementsByTagName('circle') + \
               parsed_doc.getElementsByTagName('ellipse') + parsed_doc.getElementsByTagName('line') + \
               parsed_doc.getElementsByTagName('polygon') + parsed_doc.getElementsByTagName('polyline') + \
               parsed_doc.getElementsByTagName('rect') + parsed_doc.getElementsByTagName('text')

    def create_svg_embedding(self, embedding_model="models/deepSVG_hierarchical_ordered.pth.tar"):
        """ Create SVG embedding according to deepSVG.

        Args:
            embedding model (str): Path of embedding model.

        Returns:
            torch.Tensor: SVG embedding.

        """
        return Logo._create_embedding(self.parsed_doc.toxml(), embedding_model)

    def create_path_embedding(self, embedding_model="models/deepSVG_hierarchical_ordered.pth.tar"):
        """ Create path embedding accordint to deepSVG.

        Args:
            embedding_model (str): Path of embedding model.

        Returns:
            torch.Tensor: Path embedding.

        """
        decomposed_docs = self.decompose_svg()
        embeddings = []
        for doc in decomposed_docs:
            embeddings.append(Logo._create_embedding(doc, embedding_model))
        return embeddings

    @staticmethod
    def _create_embedding(parsed_doc_xml, embedding_model):
        # The following parameters are defined in the deepSVG config:
        model_args = ['commands', 'args', 'commands', 'args']

        # The following parameters are defined in class SVGDataset:
        MAX_NUM_GROUPS = 8
        MAX_SEQ_LEN = 30
        MAX_TOTAL_LEN = 50
        PAD_VAL = -1

        deep_svg = SVG.from_str(parsed_doc_xml)
        deep_svg = Logo._simplify(deep_svg, normalize=True)
        deep_svg = Logo._numericalize(deep_svg)

        # Load pretrained model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg = Config()
        model = cfg.make_model().to(device)
        utils.load_model(embedding_model, model)
        model.eval()

        t_sep, fillings = deep_svg.to_tensor(concat_groups=False, PAD_VAL=PAD_VAL), deep_svg.to_fillings()
        # Note: DeepSVG can only handle 8 paths in a SVG and 30 sequences per path
        if len(t_sep) > 8:
            # print(f"SVG has more than 30 segments.")
            t_sep = t_sep[0:8]
            fillings = fillings[0:8]

        for i in range(len(t_sep)):
            if len(t_sep[i]) > 30:
                # print(f"Path nr {i} has more than 30 segments.")
                t_sep[i] = t_sep[i][0:30]

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
        return z

    @staticmethod
    def _simplify(deep_svg, normalize=True):
        deep_svg = deep_svg.canonicalize(normalize=normalize)
        deep_svg = deep_svg.simplify_heuristic()
        return deep_svg.normalize()

    @staticmethod
    def _numericalize(deep_svg):
        return deep_svg.numericalize(256)

    def create_df(self, pca_model="models/pca_path_embedding.sav"):
        """ Creates input data for GA and entmoot optimizer.

        Args:
            pca_model (str): Path of embedding model.

        Returns:
            pd.DataFrame: Dataframe containing filename, animation IDs, embedding, color, size and position of given logo.

        """
        filename = self.data_dir.split("/")[-1].replace(".svg", "")
        data = {'filename': filename,
                'animation_id': self.animation_ids,
                'embedding': self.create_path_embedding()}

        df = pd.DataFrame.from_dict(data)

        # Drop rows where embedding contains nan values
        df['temp'] = df['embedding'].apply(lambda row: np.isnan(row.numpy()).any())
        df = df[~df['temp']]

        # Apply PCA to embedding
        df_emb = df['embedding'].apply(lambda row: row.numpy()[0][0][0]).apply(pd.Series)
        fitted_pca = pickle5.load(open(pca_model, 'rb'))
        df_emb_red, _ = reduce_dim(df_emb, fitted_pca=fitted_pca)

        # Concatenate dataframes and drop unnecessary columns
        df = pd.concat([df, df_emb_red.reset_index(drop=True)], axis=1)
        df.drop(['temp', 'embedding'], axis=1, inplace=True)

        df['fill'] = df['animation_id'].apply(lambda row: get_style_attributes_path(self.data_dir, row, 'fill'))
        df['stroke'] = df['animation_id'].apply(lambda row: get_style_attributes_path(self.data_dir, row, 'stroke'))

        for i, c in enumerate(['r', 'g', 'b']):
            df['fill_{}'.format(c)] = df['fill'].apply(lambda row: ImageColor.getcolor(row, 'RGB')[i])

        for i, c in enumerate(['r', 'g', 'b']):
            df['stroke_{}'.format(c)] = df['stroke'].apply(lambda row: ImageColor.getcolor(row, 'RGB')[i])

        for col in ['fill_r', 'fill_g', 'fill_b', 'stroke_r', 'stroke_g', 'stroke_b']:
            df[f'svg_{col}'] = df.groupby('filename')[col].transform('mean')
            df[f'diff_{col}'] = df[col] - df[f'svg_{col}']

        df['rel_width'] = df['animation_id'].apply(lambda row: get_relative_path_size(self.data_dir, row)[0])
        df['rel_height'] = df['animation_id'].apply(lambda row: get_relative_path_size(self.data_dir, row)[1])

        df['rel_x_position'] = df['animation_id'].apply(lambda row: get_relative_path_pos(self.data_dir, row)[0])
        df['rel_y_position'] = df['animation_id'].apply(lambda row: get_relative_path_pos(self.data_dir, row)[1])
        df['nr_paths_svg'] = self.nr_paths

        df.drop(['stroke_r', 'stroke_g', 'stroke_b',
                 'svg_stroke_r', 'diff_stroke_r',
                 'svg_stroke_g', 'diff_stroke_g',
                 'svg_stroke_b', 'diff_stroke_b'], axis=1, inplace=True)

        return df

    def _insert_animation(self, animation_ids, model_output, filename_suffix=""):
        """ Function to insert multiple animation statements. """
        doc_temp = minidom.parse(self.data_dir)
        begin_values = get_begin_values_by_starting_pos(self.data_dir, animation_ids, start=1, step=0.25)
        for i, animation_id in enumerate(animation_ids):
            if not (model_output[i][:6] == np.array([0] * 6)).all():
                try:  # there are some paths that can't be embedded and don't have style attributes
                    output_dict = transform_animation_predictor_output(self.data_dir, animation_id, model_output[i])
                    output_dict["begin"] = begin_values[i]
                    if output_dict["type"] == "translate":
                        doc_temp = insert_translate_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] == "scale":
                        doc_temp = insert_scale_statement(doc_temp, animation_id, output_dict, self.data_dir)
                    if output_dict["type"] == "rotate":
                        doc_temp = insert_rotate_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] in ["skewX", "skewY"]:
                        doc_temp = insert_skew_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] == "fill":
                        doc_temp = insert_fill_statement(doc_temp, animation_id, output_dict)
                    if output_dict["type"] in ["opacity"]:
                        doc_temp = insert_opacity_statement(doc_temp, animation_id, output_dict)
                except Exception as e:
                    print(f"Logo {self.data_dir.split('/')[-1]}, animation ID {animation_id} can't be animated. {e}")
                    pass

        # Save animated SVG
        with open(f"{self.data_dir.replace('preprocessed.svg', '')}animated_{filename_suffix}.svg", 'wb') as f:
            f.write(doc_temp.toprettyxml(encoding="iso-8859-1"))