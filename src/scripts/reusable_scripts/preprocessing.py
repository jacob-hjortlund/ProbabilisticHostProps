import os
import hydra
import omegaconf
import numpy as np
import pandas as pd
import seaborn as sns

# import bayesnova.old_src.preprocessing as prep

from astropy import units as u
from astropy.table import Table
from astropy.cosmology import FLRW, Planck18
from astropy.coordinates import SkyCoord
from utils import (
    NULL_VALUE,
    PECULIAR_VELOCITY_DISPERSION,
    SPEED_OF_LIGHT,
    save_variable,
    paths,
)

default_colors = sns.color_palette("colorblind")


def identify_duplicate_sn(
    catalog: pd.pandas.DataFrame,
    max_peak_date_diff: float = 10,
    max_angular_separation: float = 1,
    sn_id_key: str = "CID",
    sn_redshift_key: str = "z",
    sn_peak_date_key: str = "PKMJD",
    sn_ra_key: str = "RA",
    sn_dec_key: str = "DEC",
) -> tuple:

    catalog = catalog.copy()
    sn_coordinates = SkyCoord(
        ra=catalog[sn_ra_key].to_numpy() * u.degree,
        dec=catalog[sn_dec_key].to_numpy() * u.degree,
        frame="icrs",
    )

    i = 0
    total_number_of_duplicate_sn = 0
    idx_of_duplicate_sn = []
    duplicate_sn_details = {}

    duplicate_subtracted_sn_coordinates = sn_coordinates.copy()
    duplicate_subtracted_catalog = catalog.copy()

    looping_condition = True
    while looping_condition:

        name_of_current_sn = duplicate_subtracted_catalog.iloc[i][sn_id_key]
        redshift_of_current_sn = duplicate_subtracted_catalog.iloc[i][sn_redshift_key]
        peak_date_of_current_sn = duplicate_subtracted_catalog.iloc[i][sn_peak_date_key]

        peak_date_diff = np.abs(
            catalog[sn_peak_date_key].to_numpy() - peak_date_of_current_sn
        )
        idx_below_max_peak_date_diff = peak_date_diff < max_peak_date_diff

        idx_below_max_angular_separation = (
            duplicate_subtracted_sn_coordinates[i].separation(sn_coordinates)
            < max_angular_separation * u.arcsec
        )

        idx_passes_match_cuts = (
            idx_below_max_peak_date_diff & idx_below_max_angular_separation
        )

        match_cids = catalog[sn_id_key][idx_passes_match_cuts].to_numpy()
        idx_match_cids = catalog[sn_id_key].isin(match_cids).to_numpy()
        no_of_missed_duplicates = idx_match_cids.sum() - idx_passes_match_cuts.sum()

        idx_match_cids[i] = False
        idx_duplicates_of_current_sn = idx_passes_match_cuts | idx_match_cids
        idx_missed_cids = idx_match_cids & ~idx_passes_match_cuts

        number_of_duplicates_for_current_sn = (
            np.count_nonzero(idx_duplicates_of_current_sn) + no_of_missed_duplicates
        )

        no_duplicates_present = number_of_duplicates_for_current_sn == 1

        if no_duplicates_present:
            i += 1
            reached_end_of_duplicate_subtracted_catalog = i == len(
                duplicate_subtracted_catalog
            )
            if reached_end_of_duplicate_subtracted_catalog:
                looping_condition = False
            continue

        total_number_of_duplicate_sn += number_of_duplicates_for_current_sn

        max_abs_peak_date_diff = np.max(
            np.abs(
                catalog[sn_peak_date_key].to_numpy()[idx_duplicates_of_current_sn]
                - peak_date_of_current_sn
            )
        )
        max_abs_redshift_diff = np.max(
            np.abs(
                catalog[sn_redshift_key].to_numpy()[idx_duplicates_of_current_sn]
                - redshift_of_current_sn
            )
        )
        names_for_duplicates = catalog[sn_id_key].to_numpy()[
            idx_duplicates_of_current_sn
        ]

        duplicate_sn_details[name_of_current_sn] = {
            "dz": max_abs_redshift_diff,
            "dt": max_abs_peak_date_diff,
            "number_of_duplicates": number_of_duplicates_for_current_sn,
            "duplicate_names": names_for_duplicates,
            "idx_duplicate": idx_duplicates_of_current_sn,
            "idx_missed_duplicates": idx_missed_cids,
            "no_of_missed_duplicates": no_of_missed_duplicates,
        }

        idx_of_duplicate_sn.append(idx_duplicates_of_current_sn)

        idx_of_non_duplicates = ~np.any(idx_of_duplicate_sn, axis=0)
        duplicate_subtracted_catalog = catalog[idx_of_non_duplicates].copy()
        duplicate_subtracted_sn_coordinates = sn_coordinates[
            idx_of_non_duplicates
        ].copy()

    return duplicate_sn_details


def filter_survey_ids(
    data: pd.DataFrame, survey_ids: list = [], survey_id_column: str = "SurveyID"
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by survey IDs.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        survey_ids (list): List of survey IDs to keep. Defaults to [], corresponding to all.
        survey_id_column (str, optional): Name of survey ID column. Defaults to "IDSURVEY".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    if len(survey_ids) == 0:
        survey_ids = tmp_data[survey_id_column].unique()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()
    idx_in_surveys = tmp_data[survey_id_column].isin(survey_ids).to_numpy()
    idx_to_keep = idx_in_surveys | idx_is_calibrator

    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_redshift(
    data: pd.DataFrame, z_min: float = 0.0, z_max: float = np.inf, z_column: str = "z"
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by redshift.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        z_min (float, optional): Minimum redshift. Defaults to 0.
        z_max (float, optional): Maximum redshift. Defaults to np.inf.
        z_column (str, optional): Name of redshift column. Defaults to "z".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()
    idx_in_z_range = (tmp_data[z_column] >= z_min) & (tmp_data[z_column] <= z_max)
    idx_to_keep = idx_in_z_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_redshift_by_survey(
    data: pd.DataFrame,
    survey_ids: list = [],
    z_min: float = 0.0,
    survey_redshift_limits: list = [],
    survey_id_column: str = "SurveyID",
    z_column: str = "z",
):
    """Filters a DataFrame by redshift for each survey.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        survey_ids (list): List of survey IDs to keep. Defaults to [], corresponding to all.
        survey_redshift_limits (dict): Dictionary of survey IDs and corresponding redshift limits.
        survey_id_column (str, optional): Name of survey ID column. Defaults to "IDSURVEY".
        z_column (str, optional): Name of redshift column. Defaults to "z".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    survey_dataframes = []
    n_removed = 0

    n_survey_ids = len(survey_ids)
    n_redshift_limits = len(survey_redshift_limits)

    if n_survey_ids != n_redshift_limits:
        raise ValueError("Length of survey IDs and redshift limits must be the same.")

    if n_survey_ids == 0:
        return tmp_data, n_removed

    for i, survey_id in enumerate(survey_ids):

        idx_survey = (tmp_data[survey_id_column] == survey_id).to_numpy()
        tmp_data_survey = tmp_data[idx_survey]
        z_max = survey_redshift_limits[i]

        tmp_data_survey, n_removed_survey = filter_redshift(
            tmp_data_survey, z_min=z_min, z_max=z_max, z_column=z_column
        )

        survey_dataframes.append(tmp_data_survey)
        n_removed += n_removed_survey

        print(f"\nSurvey ID: {survey_id}")
        print(f"Redshift range: {z_min} < z < {z_max}")
        print(f"Number of objects removed: {n_removed_survey}\n")

    tmp_data = pd.concat(survey_dataframes)

    return tmp_data, n_removed


def filter_apparent_b_mag_error(
    data: pd.DataFrame,
    b_mag_error_max: float = 0.2,
    alpha: float = 0.148,
    beta: float = 3.122,
    stretch_error_column: str = "x1Err",
    color_error_column: str = "cErr",
    redshift_column: str = "z",
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by apparent B-band magnitude error.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        b_mag_error_max (float, optional): Maximum apparent B-band magnitude error. Defaults to 0.2.
        alpha (float, optional): Alpha parameter of stretch error model. Defaults to 0.148.
        beta (float, optional): Beta parameter of stretch error model. Defaults to 3.122.
        stretch_error_column (str, optional): Name of stretch error column. Defaults to "x1Err".
        color_error_column (str, optional): Name of color error column. Defaults to "cErr".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()

    distmod_error = data[redshift_column].to_numpy() ** (-1) * (
        (5.0 / np.log(10.0)) * (PECULIAR_VELOCITY_DISPERSION / SPEED_OF_LIGHT)
    )
    apparent_b_mag_error = np.sqrt(
        distmod_error**2
        + alpha**2 * tmp_data[stretch_error_column].to_numpy() ** 2
        + beta**2 * tmp_data[color_error_column].to_numpy() ** 2
    )
    idx_in_b_mag_error_range = apparent_b_mag_error <= b_mag_error_max

    idx_to_keep = idx_in_b_mag_error_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_stretch(
    data: pd.DataFrame,
    stretch_min: float = -3.0,
    stretch_max: float = 3.0,
    stretch_column: str = "x1",
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by stretch.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        stretch_min (float, optional): Minimum stretch. Defaults to -3.
        stretch_max (float, optional): Maximum stretch. Defaults to 3.
        stretch_column (str, optional): Name of stretch column. Defaults to "x1".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()
    idx_in_stretch_range = (tmp_data[stretch_column] >= stretch_min) & (
        tmp_data[stretch_column] <= stretch_max
    )
    idx_to_keep = idx_in_stretch_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_stretch_error(
    data: pd.DataFrame,
    stretch_error_max: float = 1.5,
    stretch_error_column: str = "x1Err",
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by stretch error.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        stretch_error_max (float, optional): Maximum stretch error. Defaults to 1.5.
        stretch_error_column (str, optional): Name of stretch error column. Defaults to "x1Err".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()
    idx_in_stretch_error_range = tmp_data[stretch_error_column] <= stretch_error_max
    idx_to_keep = idx_in_stretch_error_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_color(
    data: pd.DataFrame,
    color_min: float = -0.3,
    color_max: float = 0.3,
    color_column: str = "c",
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by color.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        color_min (float, optional): Minimum color. Defaults to -0.3.
        color_max (float, optional): Maximum color. Defaults to 0.3.
        color_column (str, optional): Name of color column. Defaults to "c".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()
    idx_in_color_range = (tmp_data[color_column] >= color_min) & (
        tmp_data[color_column] <= color_max
    )
    idx_to_keep = idx_in_color_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_fitprob(
    data: pd.DataFrame, fitprob_min: float = 0.001, fitprob_column: str = "FITPROB"
):
    """Filters a DataFrame by fit probability.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        fitprob_min (float, optional): Minimum fit probability. Defaults to 0.001.
        fitprob_column (str, optional): Name of fit probability column. Defaults to "FITPROB".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()
    idx_in_fitprob_range = tmp_data[fitprob_column] >= fitprob_min
    idx_to_keep = idx_in_fitprob_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_peak_date_error(
    data: pd.DataFrame,
    peak_date_error_max: float = 2.0,
    peak_date_error_column: str = "PKMJDERR",
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by peak date error.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        peak_date_error_max (float, optional): Maximum peak date error. Defaults to 2..
        peak_date_error_column (str, optional): Name of peak date error column. Defaults to "PKMJDERR".

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()
    idx_in_peak_date_error_range = (
        tmp_data[peak_date_error_column] <= peak_date_error_max
    )
    idx_to_keep = idx_in_peak_date_error_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


def filter_tripp_residual_error(
    data: pd.DataFrame,
    tripp_residual_error_max: float = 3.5,
    intrinsic_absolute_b_mag: float = -19.253,
    intrinsic_scatter: float = 0.12,
    alpha: float = 0.148,
    beta: float = 3.122,
    apparent_b_mag_column: str = "mB",
    stretch_column: str = "x1",
    stretch_error_column: str = "x1Err",
    color_column: str = "c",
    color_error_column: str = "cErr",
    redshift_column: str = "z",
    cosmology: FLRW = Planck18,
) -> tuple[pd.DataFrame, int]:
    """Filters a DataFrame by tripp residual error. TODO: Update handling of cosmology.

    Args:
        data (pd.DataFrame): DataFrame to filter.
        tripp_residual_error_max (float, optional): Maximum tripp residual error. Defaults to 3.5.
        intrinsic_absolute_mag (float, optional): Intrinsic absolute magnitude. Defaults to -19.253.
        intrinsic_scatter (float, optional): Intrinsic scatter. Defaults to 0.12.
        alpha (float, optional): Alpha parameter. Defaults to 0.148.
        beta (float, optional): Beta parameter. Defaults to 3.122.
        apparent_b_mag_column (str, optional): Name of apparent B magnitude column. Defaults to 'mB'.
        stretch_column (str, optional): Name of stretch column. Defaults to "x1".
        stretch_error_column (str, optional): Name of stretch error column. Defaults to "x1Err".
        color_column (str, optional): Name of color column. Defaults to "c".
        color_error_column (str, optional): Name of color error column. Defaults to "cErr".
        redshift_column (str, optional): Name of redshift column. Defaults to "z".
        cosmology (FLRW, optional): Cosmology. Defaults to Planck18.

    Returns:
        tuple[pd.DataFrame, int]: Filtered DataFrame and number of removed objects.
    """

    tmp_data = data.copy()
    idx_is_calibrator = (tmp_data["is_calibrator"] == 1).to_numpy()

    distmod_error = data[redshift_column].to_numpy() ** (-1) * (
        (5.0 / np.log(10.0)) * (PECULIAR_VELOCITY_DISPERSION / SPEED_OF_LIGHT)
    )
    tripp_apparent_b_mag = (
        intrinsic_absolute_b_mag
        + cosmology.distmod(data[redshift_column].to_numpy()).value
        - alpha * data[stretch_column].to_numpy()
        + beta * data[color_column].to_numpy()
    )
    tripp_residual = data[apparent_b_mag_column].to_numpy() - tripp_apparent_b_mag
    tripp_error = np.sqrt(
        intrinsic_scatter**2
        + distmod_error**2
        + alpha**2 * data[stretch_error_column].to_numpy() ** 2
        + beta**2 * data[color_error_column].to_numpy() ** 2
    )
    idx_in_tripp_residual_error_range = (
        np.abs(tripp_residual / tripp_error) <= tripp_residual_error_max
    )

    idx_to_keep = idx_in_tripp_residual_error_range | idx_is_calibrator
    n_removed = len(tmp_data) - np.sum(idx_to_keep)

    return tmp_data[idx_to_keep], n_removed


@hydra.main(version_base=None, config_path="configs", config_name="preprocessing")
def main(cfg: omegaconf.DictConfig) -> None:

    run_name = cfg["run_name"]
    save_path = paths.data / run_name
    os.makedirs(save_path, exist_ok=True)

    # Load data

    data_path = paths.static / "data" / cfg["data"]["name"]
    data = pd.read_csv(data_path, sep=" ")

    # --------- Reformat DataFrame ---------

    # Pop Redshift Column if it Exists
    if "z" in data.columns:
        data = data.drop(columns=["z"])  # TODO: Don't hardcode this

    # Rename columns
    columns_mapping = cfg["columns_to_rename"]
    columns_to_keep = list(columns_mapping.values())
    data = data.rename(columns=columns_mapping)

    # Add New Columns
    columns_to_keep += ["duplicate_uid", "is_calibrator", "distmod_calibrator"]
    for column_name in columns_to_keep:
        if column_name not in data.columns:
            data[column_name] = NULL_VALUE
    data["HOSTGAL_sSFR"] = np.nan

    # Flag Calibrator SNe
    if cfg.get("include_calibrators", True):
        print("Including calbrator SNe...")
        calibrator_flag_column = cfg.get("calibrator_flag_column", "IS_CALIBRATOR")
        calibrator_distmod_column = cfg.get("calibrator_distmod_column", "CEPH_DIST")
        data["is_calibrator"] = (data[calibrator_flag_column].to_numpy() == 1).astype(
            float
        )
        data["distmod_calibrator"] = data[calibrator_distmod_column].to_numpy()
        n_calibrator_sn = np.sum(data["is_calibrator"].to_numpy() == 1)
        print(f"Number of calibrator SNe: {n_calibrator_sn}")

    # Format CIDs
    cids_mapping = zip(cfg["cids_to_rename"]["old"], cfg["cids_to_rename"]["new"])
    data = data.replace(cids_mapping)
    data["CID"] = data["CID"].str.lower().str.strip()

    # --------- Filter Data ---------

    filters = cfg["filters"]
    n_total_objects = len(data)
    n_filtered_objects = {}
    filtered_data = data.copy()

    for filter_name, filter_cfg in filters.items():

        print(f"\nApplying filter: {filter_name}")
        filter_func_name = f"filter_{filter_name}"
        filter_func_kwargs = filter_cfg.get("kwargs", {})
        filter_func = globals()[filter_func_name]
        filtered_data, n_removed = filter_func(filtered_data, **filter_func_kwargs)
        n_filtered_objects[filter_name] = n_removed
        print(f"Number of objects removed by {filter_name}: {n_removed}")
        print(
            f"Percentage of objects removed by {filter_name}: {n_removed / n_total_objects * 100:.2f}%"
        )
        print(f"Number of objects remaining: {len(filtered_data)}")

    # --------- Flag Duplicates ---------

    print(f"\nTotal number of objects removed: {sum(n_filtered_objects.values())}")
    print(
        f"Total percentage of objects removed: {sum(n_filtered_objects.values()) / n_total_objects * 100:.2f}%"
    )
    print(f"Total number of objects remaining: {len(filtered_data)}")

    if cfg.get("flag_duplicates", True):

        print("\nFlagging duplicates")
        duplicate_details = identify_duplicate_sn(
            filtered_data,
            max_peak_date_diff=cfg.get("max_peak_date_diff", 10),
            max_angular_separation=cfg.get("max_angular_separation", 1.0),
        )

        number_of_sn_observations = len(filtered_data)
        number_of_sn_with_duplicates = len(duplicate_details)

        dz = []
        for i, (_, sn_details) in enumerate(duplicate_details.items()):
            # number_of_duplicate_sn_observations += sn_details['number_of_duplicates']
            filtered_data.loc[sn_details["idx_duplicate"], "duplicate_uid"] = i
            dz.append(sn_details["dz"])

        n_duplicate_keys = len(duplicate_details.keys())
        if n_duplicate_keys > 0:

            duplicate_cids = np.concatenate(
                [
                    duplicate_details[key]["duplicate_names"]
                    for key in duplicate_details.keys()
                ]
            )
            unique_duplicate_cids = np.unique(duplicate_cids)
            number_of_duplicate_sn_observations = len(duplicate_cids)
        else:
            number_of_duplicate_sn_observations = 0

        number_of_unique_sn = (
            number_of_sn_observations
            - number_of_duplicate_sn_observations
            + number_of_sn_with_duplicates
        )

        for duplicate_id in range(number_of_sn_with_duplicates):
            idx_duplicate_id = filtered_data["duplicate_uid"].to_numpy() == duplicate_id
            most_common_cid = (
                filtered_data[idx_duplicate_id]["CID"].value_counts().idxmax()
            )
            filtered_data.loc[idx_duplicate_id, "CID"] = most_common_cid

        print(f"\nNumber of SNe observations in catalog: {number_of_sn_observations}")
        print(f"Number of unique SNe in catalog: {number_of_unique_sn}")
        print(
            f"Number of SNe with duplicate obsevations: {number_of_sn_with_duplicates}"
        )
        print(
            f"Total number of duplicate SNe observations: {number_of_duplicate_sn_observations}\n"
        )

        if n_duplicate_keys > 0:
            print(f"\nDuplicate SNe redshift diff statistics:")
            print(f"Max dz: {np.max(dz)}")
            print(f"Min dz: {np.min(dz)}")
            print(f"Mean dz: {np.mean(dz)}")
            print(f"Median dz: {np.median(dz)}\n")

    # --------- Cross Match sSFR ---------

    pantheon_sSFR = pd.read_csv(data_path / "Pantheon_HOSTGAL_sSFR.txt", sep=" ")
    pantheon_sSFR["CID"] = pantheon_sSFR["CID"].str.lower().str.strip()
    pantheon_sSFR = pantheon_sSFR.drop("VARNAMES:", axis=1)

    filtered_data = filtered_data.merge(pantheon_sSFR, how="left", on="CID")

    if cfg.get("cross_match_host_sSFR", True):

        print("\nCross matching w. additional catalogs for sSFRs...\n")

        uddin = Table.read(data_path / "uddin_et_al_17.fit").to_pandas()
        uddin = uddin.rename(
            columns={
                "logsSFR2": "logsSFR",
                "_RA1": "RA",
                "_DE1": "DEC",
                "zcmb1": "z",
            }
        )
        uddin["logsSFR"][uddin["logsSFR"] == 0.0] = np.nan

        jones = Table.read(data_path / "jones_et_al_18.fit").to_pandas()
        jones = jones.rename(
            columns={
                "SFR": "logsSFR",
                "_RAJ2000": "RA",
                "_DEJ2000": "DEC",
            }
        )
        jones["logsSFR"][jones["logsSFR"] == 0.0] = np.nan

        catalogs = [jones, uddin]
        catalog_names = ["jones", "uddin"]

        filtered_data_coords = SkyCoord(
            ra=filtered_data["RA"].values * u.degree,
            dec=filtered_data["DEC"].values * u.degree,
        )

        initial_number_of_sSFRs = (
            ~np.isnan(filtered_data["HOSTGAL_sSFR"].values)
        ).sum()
        initial_fraction_with_sSFR = initial_number_of_sSFRs / len(filtered_data)
        print(f"Initial number of SNe with sSFR: {initial_number_of_sSFRs}")
        print(f"Initial fraction of SNe with sSFR: {initial_fraction_with_sSFR:.2f}\n")

        for catalog, name in zip(catalogs, catalog_names):
            print(f"\nCross matching {name}...\n")

            idx_already_set = ~np.isnan(filtered_data["HOSTGAL_sSFR"].values)

            catalog_redshifts = catalog["z"].values
            catalog_coords = SkyCoord(
                ra=catalog["RA"].values * u.degree,
                dec=catalog["DEC"].values * u.degree,
            )

            idx, d2d, _ = filtered_data_coords.match_to_catalog_sky(catalog_coords)
            matched_catalog_redshifts = catalog_redshifts[idx]

            bool_idx_below_max_redshift_sep = (
                np.abs(filtered_data["zCMB"].values - matched_catalog_redshifts)
                < cfg["max_redshift_separation"]
            )
            bool_idx_match = (
                (d2d < cfg["max_angular_separation"] * u.arcsec)
                & ~idx_already_set
                & bool_idx_below_max_redshift_sep
            )
            n_matches = np.sum(bool_idx_match)
            if n_matches > 0:
                filtered_data.loc[bool_idx_match, "HOSTGAL_sSFR"] = catalog.loc[
                    idx[bool_idx_match], "logsSFR"
                ].values

            total_number_of_sSFRs = (
                ~np.isnan(filtered_data["HOSTGAL_sSFR"].values)
            ).sum()
            total_fraction_with_sSFR = total_number_of_sSFRs / len(filtered_data)
            print(f"Number of SNe matched: {n_matches}")
            print(f"Total number of SNe with sSFR: {total_number_of_sSFRs}")
            print(f"Total fraction of SNe with sSFR: {total_fraction_with_sSFR:.2f}\n")

    filtered_data["HOSTGAL_sSFR"] = filtered_data["HOSTGAL_sSFR"].fillna(NULL_VALUE)
    # --------- Save Data ---------

    filtered_data = filtered_data[columns_to_keep].copy()
    filtered_data = filtered_data.reset_index(drop=True)
    data_save_path = save_path / (cfg["run_name"] + ".dat")
    filtered_data.to_csv(data_save_path, index=False, sep=cfg["data"]["sep"])


if __name__ == "__main__":
    main()
