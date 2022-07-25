import os, glob, pickle, torch, logging
from concurrent import futures
from tqdm import tqdm
import pandas as pd
from src.preprocessing.deepsvg.svglib.svg import SVG
from src.preprocessing.deepsvg.difflib.tensor import SVGTensor
from src.preprocessing.deepsvg import utils
from src.preprocessing.deepsvg.svglib.geom import Bbox
from src.preprocessing.deepsvg.svg_dataset import load_dataset  # SVG Dataset
from src.preprocessing.deepsvg.utils.utils import batchify
from src.preprocessing.configs.deepsvg.hierarchical_ordered import Config

# Reproducibility
utils.set_seed(42)

# Define path to model
model_path = "models/deepSVG_hierarchical_ordered.pth.tar"


def _load_model_and_dataset(data_folder="data/svgs"):
    # Load pretrained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = cfg.make_model().to(device)
    utils.load_model(model_path, model)
    model.eval();

    # Load dataset
    cfg.data_dir = f"{data_folder}/"
    cfg.meta_filepath = f"data/meta_data/{data_folder.split('/')[-1]}_meta.csv"
    dataset = load_dataset(cfg)
    return dataset, model, device, cfg


def encode_svg(filename, data_folder="data/svgs", split_paths=True):
    """ Get embedding of given SVG.

    Args:
        filename (str): Path of SVG.
        data_folder (str): If data_folder="data/svgs", SVG embeddings are returned.
                            If data_folder="data/decomposed_svgs", path embeddings are returned.
        split_paths (bool): If true, additional preprocessing step is carried out, where paths consisting of multiple
                            paths are split into multiple paths.

    Returns:
        torch.Tensor: SVG embedding.

    """
    dataset, model, device, cfg = _load_model_and_dataset(data_folder=data_folder)
    return _encode_svg(dataset, filename, model, device, cfg, split_paths)


def decode_z(z, data_folder="data/svgs", do_display=True, return_svg=False):
    """ Decode given SVG embedding.

    Args:
        z (str): SVG embedding.
        data_folder (str): Path of folder containing all SVGs.
        do_display (bool): If true, SVG is displayed.
        return_svg (bool): If true, SVG is returned.

    Returns:
        deepsvg.svglib.svg.SVG: Decoded SVG embedding.

    """
    dataset, model, device, cfg = _load_model_and_dataset(data_folder=data_folder)
    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256),
                                      allow_empty=True).normalize().split_paths().set_color("random")

    if return_svg:
        return svg_path_sample

    return svg_path_sample.draw(do_display=do_display, return_png=False)


def apply_embedding_model_to_svgs(data_folder="data/svgs", split_paths=True, save=True):
    """ Get embeddings of all SVGs in a given folder.

    Args:
        data_folder (str): If data_folder="data/svgs", SVG embeddings are returned.
                            If data_folder="data/decomposed_svgs", path embeddings are returned.
        split_paths (bool): If true, additional preprocessing step is carried out, where paths consisting of multiple
                            paths are split into multiple paths.
        save (bool): If true, SVG embedding is saved as pd.DataFrame in folder data/embeddings.

    Returns:
        pd.DataFrame: Dataframe containing filename, animation_id (if data_folder="decomposed_svgs") and embedding.

    """
    dataset, model, device, cfg = _load_model_and_dataset(data_folder=data_folder)
    cfg.data_dir = data_folder

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        svg_files = glob.glob(os.path.join(cfg.data_dir, "*.svg"))
        svg_list = []
        with tqdm(total=len(svg_files)) as pbar:
            embedding_requests = [
                executor.submit(_apply_embedding_model_to_svg, dataset, svg_file, svg_list, model, device, cfg,
                                split_paths)
                for svg_file in svg_files]

            for _ in futures.as_completed(embedding_requests):
                pbar.update(1)

    df = pd.DataFrame.from_records(svg_list, index='filename')['embedding'].apply(pd.Series)
    df.reset_index(level=0, inplace=True)

    if data_folder == "data/decomposed_svgs":
        df['animation_id'] = df['filename'].apply(lambda row: row.split('_')[-1])
        cols = list(df.columns)
        cols = [cols[0], cols[-1]] + cols[1:-1]
        df = df.reindex(columns=cols)
        df['filename'] = df['filename'].apply(lambda row: "_".join(row.split('_')[0:-1]))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if save:
        model = model_path.split("/")[-1].replace("_svgs", "").replace("_decomposed", "").replace(".pth.tar", "")
        data = data_folder.split("/")[-1]
        output = open(f'data/embeddings/{model}_{data}_embedding.pkl', 'wb')
        pickle.dump(df, output)
        output.close()

    logging.info("Embedding complete.")

    return df


def _apply_embedding_model_to_svg(dataset, svg_file, svg_list, model, device, cfg, split_paths):
    z = _encode_svg(dataset, svg_file, model, device, cfg, split_paths).numpy()[0][0][0]
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    dict_data = {"filename": filename,
                 "embedding": z}

    svg_list.append(dict_data)


def _encode_svg(dataset, filename, model, device, cfg, split_paths):
    # Note: Only 30 segments per path are allowed. Paths are cut after the first 30 segments.
    svg = SVG.load_svg(filename)
    if split_paths:
        svg = dataset.simplify(svg)
        svg = dataset.preprocess(svg, augment=False)
        data = dataset.get(svg=svg)
    else:  # Here: paths are not split
        svg = _canonicalize_without_path_splitting(svg, normalize=True)
        svg = dataset.preprocess(svg, augment=False)
        data = dataset.get(svg=svg)

    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z


def _canonicalize_without_path_splitting(svg, normalize=False):
    svg.to_path().simplify_arcs()
    if normalize:
        svg.normalize()
    svg.filter_consecutives()
    svg.filter_empty()
    svg._apply_to_paths("reorder")
    svg.svg_path_groups = sorted(svg.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
    svg._apply_to_paths("canonicalize")
    svg.recompute_origins()
    svg.drop_z()
    return svg
