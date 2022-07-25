""" This function aggregates the path decisions labels for training.
"""

import pandas as pd


def aggregate_path_animation_decisions_label(path_decision_labels_dataset=
                                             "../../data/interim/logos_paths_animation_decision_label"
                                             "/logos_dataset_paths_animation_decisions.csv",
                                             matching_filenames_canva_dataset="../../data/interim"
                                                                              "/logos_paths_information "
                                                                              "/matching_filenames_canva.csv",
                                             matching_filenames_designer_dataset="../../data/interim"
                                                                                 "/logos_paths_information"
                                                                                 "/matching_filenames_designer.csv"):
    """ Function to aggregate path decisions labels.

    Args:
        matching_filenames_designer_dataset (string): dataset to match designer filenames to logo ids
        matching_filenames_canva_dataset (string): dataset to match canva filenames to logo ids
        path_decision_labels_dataset (string): path to path decisions label dataset.

    Returns:
        pd.DataFrame: Dataframe containing names of scraped logos.

    """
    path_decision_labels = pd.read_csv(path_decision_labels_dataset,
                                       skiprows=[1],
                                       dtype={"logo": object, "path_0": bool, "path_1": bool, "path_2": bool,
                                              "path_3": bool, "path_4": bool,
                                              "path_5": bool, "path_6": bool, "path_7": bool, "alias": object})

    path_decision_labels_id = path_decision_labels.assign(
        logo_id=path_decision_labels.logo.str.extract("(\d+)")).astype(
        {"logo_id": 'int64'})
    path_decision_labels_melt = pd.melt(path_decision_labels_id, id_vars=["logo", "alias", "logo_id"],
                                        value_vars=["path_0", "path_1", "path_2", "path_3", "path_4", "path_5",
                                                    "path_6",
                                                    "path_7"],
                                        var_name="path", value_name="animate")
    path_decision_labels_melt_type = path_decision_labels_melt.assign(
        order_id=path_decision_labels_melt.path.str.extract("(\d+)")).astype({"order_id": 'int64'})

    data_matching_filenames_canva = pd.read_csv(
        matching_filenames_canva_dataset)
    data_matching_filename_designer = pd.read_csv(matching_filenames_designer_dataset)

    data_matching_filename_canva_clean = data_matching_filenames_canva.assign(
        filename=data_matching_filenames_canva.logo_id,
        logo_id=data_matching_filenames_canva.logo_id.str.extract("(\d+)"))

    data_matching_filename_id = pd.concat([data_matching_filename_designer,
                                           data_matching_filename_canva_clean]).astype({"logo_id": 'int64'})

    label_path_merged = pd.merge(path_decision_labels_melt_type, data_matching_filename_id, how='left',
                                 on=["logo_id", "order_id"])

    # filter data for trusted persons
    trusted_aliases = ["Jani", "Jakob", "Jonathan", "Julia", "Kikipu", "Lena",
                       "Niklas", "Ramo", "rebecca", "Sarah_240321", "Tim_Harry",
                       "Jan B.", "Sarah", "Tim"]
    label_path_merged_filter = label_path_merged[label_path_merged.alias.isin(trusted_aliases)].dropna()

    path_label_final = label_path_merged_filter.groupby(by=["filename", "logo_id", "order_id", "animation_id"]).mean(
        "animate")

    path_label_final.to_csv("../../data/processed/logos_dataset_path_decisions_label"
                            "/logos_dataset_path_decisions_label.csv")
