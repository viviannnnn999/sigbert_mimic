########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                     PACKAGES IMPORT                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


import pandas as pd
from typing import Optional
import numpy as np
# %matplotlib inline
import time


# Package COMPRESSION ###################
from compression_pkg import apply_linear_projection

# Package SIGNATURES ###################
from signature_mimic import preprocess_time, signature_extract, preprocess_sign

# Package SURVIVAL ANALYSIS ###################
from survival_analysis_sigbert import preprocess_cox, feat_event_extract, global_cox_train, skglm_datatest


# Package SKLEARN
import warnings
# set_config(display="text")  # displays text representation of estimators
warnings.filterwarnings("ignore", 
                        message="invalid value encountered in divide", 
                        category=RuntimeWarning)


from sklearn.model_selection import train_test_split





import matplotlib.pyplot as plt
import seaborn as sns

def print_dataset_statistics(df, var_id, var_death):
    """
    Computes and prints basic descriptive statistics about the survival dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns:
        - 'ID' : patient identifier
        - 'DEATH' : 1 if the patient is deceased, 0 if censored

    Prints
    ------
    - Total number of patients
    - Total number of medical reports
    - Average number of reports per patient
    - Number of deceased patients
    - Number of censored patients
    """
    # Compute statistics
    num_unique_patients = df[var_id].nunique()
    num_total_reports = len(df)
    mean_reports_per_patient = df.groupby(var_id).size().mean()
    num_deceased = df[df[var_death] == 1][var_id].nunique()
    num_censored = df[df[var_death] == 0][var_id].nunique()

    # Print results
    print(f"Total number of patients in the dataset: {num_unique_patients}")
    print(f"Total number of medical reports: {num_total_reports}")
    print(f"Average number of reports per patient: {mean_reports_per_patient:.2f}")
    print(f"Number of deceased patients: {num_deceased}")
    print(f"Number of censored patients: {num_censored}")


def plot_report_distribution_per_patient(df, var_id, export_path=None):
    """
    Plots the distribution of the number of reports per patient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column 'ID' indicating patient identifiers.
    export_path : str or None, default=None
        If provided, saves the plot to this path as a PNG file.
    """
    report_counts = df[var_id].value_counts()

    plt.figure(figsize=(10, 6))
    sns.histplot(report_counts, bins=20, kde=True, color='green')

    plt.xlabel("Number of reports per patient", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.title("Distribution of number of reports per patient", fontsize=15)
    plt.grid(True)

    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')

    plt.show()




########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                     DATA PREPROCESS                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


def convert_date_columns(data, verbose=True, date_format="%Y-%m-%d"):
    """
    Automatically convert columns that likely contain dates into pandas datetime format.

    This function scans the column names of the DataFrame and identifies any columns
    starting with 'DATE', 'Date_', or 'date_' as potential date fields. It then attempts
    to parse their values using `pd.to_datetime` with the specified date format, coercing
    invalid entries to NaT.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame potentially containing date-like columns.
    verbose : bool, default=True
        Whether to print messages when missing values (NaT) are detected after conversion.
    date_format : str, default="%Y-%m-%d"
        Expected date format for parsing (e.g. ISO format: year-month-day).

    Returns
    -------
    pd.DataFrame
        The original DataFrame with specified columns converted to datetime format.
    """
    date_columns = [col for col in data.columns if col.startswith('DATE')
                    or col.startswith('Date_') or col.startswith('date_')]

    for col in date_columns:
        data[col] = pd.to_datetime(data[col], format=date_format, errors='coerce')
        if data[col].isna().any() and verbose:
            print(f"Warning: Column '{col}' contains NaT values after conversion with format '{date_format}'.")

    return data





def prep_import(
    df_longitudinal,
    signature_order=3,
    use_log_signature=False,
    use_levy_area=False,
    print_progress=False,
    patient_id_col="ID",
    embedding_col="embeddings",
    event_col="DEATH",
    duration_col="duration",
    survival_time_col="time",
    survival_event_col="event",
    verbose=True
):
    """
    Prepare longitudinal clinical data for survival analysis using
    path-signature features extracted from sequential embeddings.

    This function performs:
    - temporal normalization,
    - path signature extraction,
    - signature preprocessing,
    - Cox-compatible feature extraction,
    - survival outcome formatting.

    Parameters
    ----------
    df_longitudinal : pd.DataFrame
        Longitudinal dataset containing sequential clinical reports.
    signature_order : int, default=3
        Truncation order of the path signature.
    use_log_signature : bool, default=False
        Whether to compute log-signatures instead of classical signatures.
    use_levy_area : bool, default=False
        Whether to include Lévy area features.
    print_progress : bool, default=False
        Whether to print intermediate progress messages.
    patient_id_col : str, default="ID"
        Column identifying patients.
    embedding_col : str, default="embeddings"
        Column containing embedding vectors.
    event_col : str, default="DEATH"
        Binary event indicator column.
    duration_col : str, default="duration"
        Survival duration column.
    survival_time_col : str, default="time"
        Name of the output survival time column.
    survival_event_col : str, default="event"
        Name of the output survival event column.
    verbose : bool, default=True
        Whether to enable verbose mode during signature extraction.

    Returns
    -------
    Xt : np.ndarray
        Signature feature matrix.
    y : np.ndarray
        Structured survival outcome array.
    feature_names : list
        Names of signature features.
    nbr_signature_features : int
        Number of signature coefficients.
    nbr_levy_features : int
        Number of Lévy area features.
    patient_ids : np.ndarray
        Patient identifiers retained in the final dataset.
    df_study : pd.DataFrame
        Final dataframe combining features and survival outcomes.
    """

    start_time = time.time()

    df = df_longitudinal.copy()

    # --------------------------------------------------
    # Normalize timestamps between 0 and 1
    # --------------------------------------------------
    df_time_normalized = preprocess_time(
        df,
        patient_id_col=patient_id_col
    )

    if print_progress:
        print("*" * 30)
        print("Time normalization completed.")

    # --------------------------------------------------
    # Signature extraction
    # --------------------------------------------------
    df_signature, nbr_signature_features, nbr_levy_features = (
        signature_extract(
            df_time_normalized,
            order=signature_order,
            var_patient=patient_id_col,
            var_embd=embedding_col,
            use_log=use_log_signature,
            use_mat_Levy=use_levy_area,
            verbose=verbose
        )
    )

    if print_progress:
        print("*" * 30)
        print("Signature extraction completed.")

    # --------------------------------------------------
    # Signature preprocessing
    # --------------------------------------------------
    df_signature = preprocess_sign(
        df_signature,
        retire_small=False,
        return_id=False
    )

    if print_progress:
        print("*" * 30)
        print("Signature preprocessing completed.")

    # --------------------------------------------------
    # Cox-compatible preprocessing
    # --------------------------------------------------
    df_signature_filtered, feature_names, patient_ids = (
        preprocess_cox(
            df_signature,
            return_id=True
        )
    )

    # --------------------------------------------------
    # Final feature extraction
    # --------------------------------------------------
    Xt, y, patient_ids = feat_event_extract(
        df_signature_filtered,
        features=feature_names,
        var_id=patient_id_col,
        var_DEATH=event_col,
        var_duree=duration_col
    )

    if print_progress:
        print("Survival preprocessing completed.")
        print(
            f"Total preprocessing time: "
            f"{time.time() - start_time:.2f} seconds"
        )
        print("-" * 70)

    # --------------------------------------------------
    # Final study dataframe
    # --------------------------------------------------
    df_study = pd.DataFrame(
        Xt,
        columns=feature_names
    )

    df_study.insert(
        0,
        patient_id_col,
        patient_ids
    )

    df_study[survival_event_col] = (
        y[survival_event_col]
        .astype(bool)
    )

    df_study[survival_time_col] = (
        y[survival_time_col]
        .astype(float)
    )

    if print_progress:

        print(
            f"Final study dataframe shape: "
            f"{df_study.shape}"
        )

    return (
        Xt,
        y,
        feature_names,
        nbr_signature_features,
        nbr_levy_features,
        patient_ids,
        df_study
    )
    

def make_train_test(
    df_OG, 
    var_id='ID', 
    var_date='date_creation', 
    min_date='1990-01-01', 
    n_group=10, 
    random_state=177, 
    size_test=0.5,
    verbose=False
):
    """
    Splits a DataFrame into a training set and multiple test sets without balancing for survival status.

    Parameters
    ----------
    df_OG : pd.DataFrame
        The full dataset containing patient-level data, including identifier and date columns.
    var_id : str, default='ID'
        Name of the column identifying each patient.
    var_date : str, default='date_creation'
        Name of the column containing the date of the medical report.
    min_date : str, default='1990-01-01'
        Minimum accepted date for filtering patient records.
    n_group : int, default=10
        Number of groups to split the test set into.
    random_state : int, default=177
        Random seed for reproducibility.
    size_test : float, default=0.5
        Proportion of patients to include in the test set.
    verbose : bool, default=False
        Whether to print detailed information about the split.

    Returns
    -------
    tuple
        - df_train_new : pd.DataFrame
            The unbalanced training set.
        - test_groups : list of pd.DataFrame
            List of test sets split into `n_group` subsets.
    """
    df = df_OG.copy()

    # Shuffle within each patient group
    df = df.groupby(var_id, group_keys=False).apply(
        lambda x: x.sample(frac=1, random_state=random_state)
    ).reset_index(drop=True)

    # Filter patients with all dates before min_date
    min_date = pd.to_datetime(min_date)
    df = df[df.groupby(var_id)[var_date].transform('min') > min_date]

    # Unique patient IDs
    unique_ids = df[var_id].unique()

    # Train/test split
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=size_test, random_state=random_state
    )

    df_train_new = df[df[var_id].isin(train_ids)]
    df_test_combined = df[df[var_id].isin(test_ids)]

    # Split test set into n groups
    test_ids_splits = np.array_split(test_ids, n_group)
    test_groups = [
        df_test_combined[df_test_combined[var_id].isin(split)]
        for split in test_ids_splits
    ]

    if verbose:
        print(f"Number of unique patients in training set: {df_train_new[var_id].nunique()}")
        for i, group in enumerate(test_groups, start=1):
            print(f"Number of unique patients in test group {i}: {group[var_id].nunique()}")

    return df_train_new, test_groups


    

def make_df_conform(
    df: pd.DataFrame,
    var_id: str = 'ID',
    var_start: str = 'date_start',
    var_end: str = 'date_end',
    var_death: str = 'DEATH',
    var_death_date: str = 'date_death',
    var_T: str = 'T_days',
    var_known: Optional[str] = None,     # e.g. 'duration_known'
    var_gap: str = 'death_know_gap',
    limite_gap: Optional[int] = None,    # ← FIXED here
    verbose: bool = True
):
    """
    Conform dataset for survival analysis using patient-level T_days.
    Returns (df_filtered, df_last_obs_one_row_per_patient, id_list).
    """
    # Ensure datetime
    for col in [var_start, var_end, var_death_date]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # If T_days missing, compute once from last row per patient
    if var_T not in df.columns:
        if verbose:
            print(f"[make_df_conform] '{var_T}' not found — computing from dates.")
        df_last_obs_tmp = (
            df.sort_values(by=var_end)
              .groupby(var_id, as_index=False)
              .last()
              .loc[:, [var_id, var_death, var_start, var_end, var_death_date]]
              .assign(**{
                  var_T: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )
              })
        )
        df = df.merge(df_last_obs_tmp[[var_id, var_T]], on=var_id, how='left')

    # Build one-line-per-patient view
    keep_cols = [c for c in [var_id, var_death, var_start, var_end, var_death_date, var_T, var_known] if c]
    df_last_obs = (
        df.sort_values(by=var_end)
          .groupby(var_id, as_index=False)
          .last()[keep_cols]
          .copy()
    )

    # Compute death gap only for deceased
    if var_known:
        if var_known not in df_last_obs.columns:
            raise ValueError(f"'{var_known}' not found in DataFrame.")
        df_last_obs[var_gap] = np.where(
            df_last_obs[var_death] == 1,
            (df_last_obs[var_death_date] - (df_last_obs[var_start] + pd.to_timedelta(df_last_obs[var_known], unit='d'))).dt.days,
            np.nan
        )

    # Filter invalid cases
    df_last_obs = (
        df_last_obs
        .dropna(subset=[var_start, var_end, var_T])
        .loc[df_last_obs[var_T] > 0]
        .reset_index(drop=True)
    )
    id_list = df_last_obs[var_id].values

    # Filter df accordingly
    initial_lines = len(df)
    initial_ids = df[var_id].nunique()
    df = df[df[var_id].isin(id_list)].copy()
    final_lines = len(df)
    final_ids = df[var_id].nunique()

    if verbose:
        print(f"Removed rows: {initial_lines - final_lines}")
        print(f"Removed patient IDs: {initial_ids - final_ids}")
        kept_prop = (final_ids / initial_ids) if initial_ids else 0.0
        print(f"Kept patient IDs: {final_ids}/{initial_ids} ({kept_prop:.1%})")

    # Optional: filter by death gap threshold
    if limite_gap is not None and var_known:
        pre_gap_n = df_last_obs.shape[0]
        df_last_obs = df_last_obs[df_last_obs[var_gap] <= limite_gap].copy()
        removed_gap = pre_gap_n - df_last_obs.shape[0]
        id_list = df_last_obs[var_id].values
        df = df[df[var_id].isin(id_list)].copy()
        if verbose:
            print(f"Removed by {var_gap} ≤ {limite_gap}: {removed_gap} patients")

    return df, df_last_obs, id_list



def global_sigbert_mimic_pipeline(
    df_full,
    df_train,
    test_sets,
    projection_matrix,
    lambda_l1=0.7,
    signature_order=2,
    use_levy_area=False,
    print_progress=False,
    patient_id_col="SUBJECT_ID",
    embedding_col="embeddings",
    timestamp_col="note_time",
    event_col="delta_i",
    duration_col="survival_time",
    cox_model_type="sk_cox",
    use_standard_scaling=False
):
    """
    End-to-end SigBERT pipeline applied to longitudinal clinical data.

    This pipeline:
    - formats longitudinal survival data,
    - projects embeddings into a compressed space,
    - computes path-signature features,
    - trains a Cox survival model,
    - evaluates performance on multiple test sets.

    Parameters
    ----------
    df_full : pd.DataFrame
        Complete longitudinal dataset.
    df_train : pd.DataFrame
        Training dataframe.
    test_sets : list
        List of held-out test dataframes.
    projection_matrix : np.ndarray
        Linear projection matrix applied to embeddings.
    lambda_l1 : float, default=0.7
        L1 regularization strength for the Cox model.
    signature_order : int, default=2
        Truncation order of the path signature.
    use_levy_area : bool, default=False
        Whether to include Lévy area features.
    use_standard_scaling : bool, default=False
        Whether to standardize features before training.

    Returns
    -------
    tuple
        Training outputs, survival predictions and evaluation metrics.
    """

    from sklearn.preprocessing import StandardScaler

    print("\n### Running SigBERT pipeline on MIMIC dataset ###")

    time_start = time.time()

    # --------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------
    reports_per_patient = (
        df_full.groupby(patient_id_col)[timestamp_col]
               .count()
    )

    std_reports_per_patient = reports_per_patient.std()

    # --------------------------------------------------
    # Train formatting
    # --------------------------------------------------
    df_train_processed, df_train_last_obs, train_patient_ids = (
        make_df_conform(
            df_train,
            var_id=patient_id_col,
            var_T=duration_col,
            verbose=False
        )
    )

    total_train_patients = (
        df_train_last_obs[patient_id_col]
        .nunique()
    )

    print(
        f"\nTotal number of patients in train set: "
        f"{total_train_patients}"
    )

    # --------------------------------------------------
    # Test formatting
    # --------------------------------------------------
    processed_test_sets = []
    all_test_last_obs = []

    for df_test in test_sets:

        df_test_processed, df_test_last_obs, _ = (
            make_df_conform(
                df_test,
                var_id=patient_id_col,
                var_T=duration_col,
                verbose=False
            )
        )

        processed_test_sets.append(df_test_processed)
        all_test_last_obs.append(df_test_last_obs)

    df_test_all = pd.concat(
        processed_test_sets,
        axis=0
    )

    df_test_last_obs_all = pd.concat(
        all_test_last_obs,
        axis=0
    )

    total_test_patients = (
        df_test_last_obs_all[patient_id_col]
        .nunique()
    )

    print(
        f"Total number of patients in validation set: "
        f"{total_test_patients}\n"
    )

    # --------------------------------------------------
    # Global dataset statistics
    # --------------------------------------------------
    df_all_processed = pd.concat(
        [df_train_processed, df_test_all],
        axis=0
    )

    total_unique_patients = (
        df_all_processed[patient_id_col]
        .nunique()
    )

    total_reports = len(df_all_processed)

    mean_reports_per_patient = (
        df_all_processed.groupby(patient_id_col)
                        .size()
                        .mean()
    )

    total_deceased_patients = (
        df_all_processed[
            df_all_processed[event_col] == 1
        ][patient_id_col].nunique()
    )

    total_censored_patients = (
        df_all_processed[
            df_all_processed[event_col] == 0
        ][patient_id_col].nunique()
    )

    # --------------------------------------------------
    # Study duration statistics
    # --------------------------------------------------
    df_all_last_obs = pd.concat(
        [df_train_last_obs, df_test_last_obs_all]
    )

    mean_study_time = np.mean(
        df_all_last_obs[duration_col]
    )

    std_study_time = np.std(
        df_all_last_obs[duration_col]
    )

    # --------------------------------------------------
    # Embedding projection
    # --------------------------------------------------
    df_train_projected = apply_linear_projection(
        df_train_processed,
        projection_matrix,
        var_embd=embedding_col
    )

    # --------------------------------------------------
    # Signature extraction
    # --------------------------------------------------
    Xt_train, y_train, feature_names, nbr_sig, nbr_levy, train_ids, df_study_train = (
        prep_import(
            df_train_projected,
            t_pred=None,
            order_sign=signature_order,
            use_mat_Levy=use_levy_area,
            print_progress=print_progress
        )
    )

    # --------------------------------------------------
    # Optional scaling
    # --------------------------------------------------
    scaler = None

    if use_standard_scaling:

        scaler = StandardScaler()

        Xt_train = scaler.fit_transform(
            Xt_train
        )

    # --------------------------------------------------
    # Cox model training
    # --------------------------------------------------
    print("\n--------------- Cox training ---------------")

    cph, df_survival_train, w_cox, scores, X_train, y_cox, c_index_train, log_likelihood, _ = (
        global_cox_train(
            Xt_train,
            y_train,
            id_list_train=train_ids,
            learning_cox_map=cox_model_type,
            lambda_l1_CV=lambda_l1
        )
    )

    print("--------------------------------------------")

    # --------------------------------------------------
    # Test evaluation
    # --------------------------------------------------
    c_index_test_list = []
    survival_test_outputs = []

    for i, df_test_processed in enumerate(processed_test_sets, start=1):

        df_test_projected = apply_linear_projection(
            df_test_processed,
            projection_matrix,
            var_embd=embedding_col
        )

        Xt_test, y_test, _, _, _, test_ids, df_study_test = (
            prep_import(
                df_test_projected,
                t_pred=None,
                order_sign=signature_order,
                use_mat_Levy=use_levy_area,
                print_progress=print_progress
            )
        )

        if use_standard_scaling:
            Xt_test = scaler.transform(Xt_test)

        print(f"\n--- Evaluation on test split {i} ---")

        df_survival_test, c_index_test, X_test, y_test_final = (
            skglm_datatest(
                Xt_test,
                y_test,
                w_cox,
                cph,
                test_ids,
                plot_curves=False
            )
        )

        c_index_test_list.append(
            c_index_test
        )

        survival_test_outputs.append(
            df_survival_test
        )

    # --------------------------------------------------
    # Final statistics
    # --------------------------------------------------
    c_index_test_mean = np.mean(
        c_index_test_list
    )

    c_index_test_std = np.std(
        c_index_test_list,
        ddof=1
    )

    total_runtime = time.time() - time_start

    print(f"\nMean test C-index: {c_index_test_mean:.4f}")
    print(f"Std test C-index: {c_index_test_std:.4f}")

    # --------------------------------------------------
    # Summary dataframe
    # --------------------------------------------------
    df_results = pd.DataFrame([{
        "Mean C-index": c_index_test_mean,
        "Std C-index": c_index_test_std,
        "Total Deceased Patients": total_deceased_patients,
        "Total Censored Patients": total_censored_patients,
        "Total Unique Patients": total_unique_patients,
        "Total Number of Reports": total_reports,
        "Mean Study Time (days)": round(mean_study_time, 3),
        "Std Study Time (days)": round(std_study_time, 3),
        "Execution Time (s)": round(total_runtime, 2),
        "Mean Reports per Patient": round(mean_reports_per_patient, 3),
        "Std Reports per Patient": round(std_reports_per_patient, 3),
    }])

    return (
        df_results,
        cph,
        df_survival_train,
        w_cox,
        scores,
        X_train,
        y_train,
        y_cox,
        c_index_train,
        c_index_test_list,
        c_index_test_mean,
        c_index_test_std,
        survival_test_outputs
    )
    
    
########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                   LANDMARK APPROACH                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################



def compute_survival_time(df: pd.DataFrame,
                          var_id: str = 'ID',
                          var_death: str = 'DEATH',
                          var_start: str = 'date_start',
                          var_end: str = 'date_end',
                          var_death_date: str = 'date_death',
                          new_var_time: str = 'T_days',
                          verbose: bool = True):
    """
    Compute survival time T_i = (date_death - date_start) if death occurred,
    else (date_end - date_start). The result is added to df for each patient's
    observations and also returned as a patient-level summary.
    """
    # Ensure datetime format
    for col in [var_start, var_end, var_death_date]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute T_i for each patient (one row per ID)
    df_surv = (
        df.sort_values(by=var_end)
          .groupby(var_id, as_index=False)
          .last()
          .loc[:, [var_id, var_death, var_start, var_end, var_death_date]]
          .assign(
              **{
                  new_var_time: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )
              }
          )
    )

    # Merge survival times back to full df
    df = df.merge(df_surv[[var_id, new_var_time]], on=var_id, how='left')

    if verbose:
        print(f"[compute_survival_time] {df_surv.shape[0]} patients processed.")
        print(f"  Mean T: {df_surv[new_var_time].mean():.1f} days "
              f"({df_surv[new_var_time].std():.1f} SD)")
        print(f"  Deaths: {df_surv[var_death].sum()} / {df_surv.shape[0]}")

    return df, df_surv



def define_landmark_cohort(df: pd.DataFrame,
                           landmark_months: int,
                           var_time: str = 'date_creation',
                           var_id: str = 'ID',
                           var_start: str = 'date_start',
                           var_end: str = 'date_end',
                           var_T: str = 'T_days',
                           window_months: int = 6,
                           verbose: bool = True):
    """
    Build the landmark cohort for a given relative time (in months) since baseline.
    Patients with T_days >= L_days are kept, reports restricted to [L-w, L].
    Returns:
        - df_L : sequential data restricted to [L-w, L]
        - patients_in_study : list of eligible IDs
        - df_gamma : one row per patient with gamma_i(L)
    """

    # Ensure datetime types
    for col in [var_time, var_start, var_end]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert months to days
    L_days = int(landmark_months * 30)
    w_days = int(window_months * 30)

    # Patients still in study at the landmark: T_days >= L_days
    patients_in_study = df.loc[df[var_T] >= L_days, var_id].unique().tolist()

    # Subset: keep only patients at risk
    df_sub = df[df[var_id].isin(patients_in_study)].copy()

    # Compute days since start
    df_sub['days_since_start'] = (df_sub[var_time] - df_sub[var_start]).dt.days

    # Apply time window [L-w, L]
    total_obs_before = len(df_sub)
    df_L = df_sub[(df_sub['days_since_start'] >= (L_days - w_days)) &
                  (df_sub['days_since_start'] <= L_days)].copy()
    total_obs_after = len(df_L)
    obs_removed = total_obs_before - total_obs_after
    prop_removed = obs_removed / total_obs_before if total_obs_before > 0 else 0

    # ---------------------------------------------------------------
    # Compute gamma_i(L), but DO NOT MERGE into sequential dataframe
    # ---------------------------------------------------------------
    first_obs = (
        df_L.groupby(var_id)['days_since_start']
        .min()
        .reset_index()
        .rename(columns={'days_since_start': 'first_obs_days'})
    )
    first_obs['gamma'] = ((L_days - first_obs['first_obs_days']) < w_days).astype(int)

    # Prepare gamma output: only ID and gamma
    df_gamma = first_obs[[var_id, 'gamma']].copy()
    # ---------------------------------------------------------------

    # ===============================================================
    # === IMPORTANT LANDMARK FIX: UPDATE RESIDUAL SURVIVAL TIME R ===
    # ===============================================================
    df_L["R"] = df_L[var_T] - L_days
    # ===============================================================

    # ====================================================================
    # === SECOND IMPORTANT FIX: UPDATE EVENT INDICATOR FOR LANDMARKING ===
    # === δ_i(L) = 1 if patient dies after L; 0 otherwise                ===
    # ====================================================================
    df_L["DEATH_L"] = (
        (df_L[var_T] > L_days) &
        (df_L["DEATH"] == 1)
    ).astype(int)
    # ====================================================================

    # --- Verbose summary ---
    if verbose:
        n_total = df[var_id].nunique()
        n_kept = len(patients_in_study)
        n_gamma = df_gamma['gamma'].sum()
        prop_gamma = n_gamma / n_kept if n_kept > 0 else 0

        print(f"[Landmark {landmark_months} mo] Patients kept: {n_kept}/{n_total} "
              f"({n_total - n_kept} excluded)")
        print(f"  → Short-history γ=1: {n_gamma}/{n_kept} = {prop_gamma:.1%}")
        print(f"  → Observations kept: {total_obs_after}/{total_obs_before} "
              f"({100 * (1 - prop_removed):.1f}% retained, {100 * prop_removed:.1f}% removed)")
        print(f"  → Residual survival R_i computed (min={df_L['R'].min()} days).")
        print(f"  → Landmark events computed: DEATH_L sum = {df_L['DEATH_L'].sum()}.")

    return df_L, patients_in_study, df_gamma




# UPDATE 2026




def prep_signature_cox(
    df_OG,
    order_sign=2,
    use_log=False,
    use_mat_Levy=False,
    print_progress=False,
    var_id="ID",
    var_embd="embeddings",
    var_start="date_start",
    var_death="DEATH_L",
    var_duration="R",
    var_time="time",
    var_event="event",
    interpolation_type=None,
    var_struct_seq_list_OG=None,
    verbose=True
):
    """
    Prepare Cox-ready data using path-signature features extracted
    from sequential embeddings (landmark setting).
    """

    import time
    start = time.time()

    df = df_OG.copy()

    # ------------------------------------------------------------------
    # 1. Time normalization
    # ------------------------------------------------------------------
    df_time = preprocess_time(df)

    if print_progress:
        print("Time normalization completed.")

    # ------------------------------------------------------------------
    # 2. Signature extraction (landmark version)
    # ------------------------------------------------------------------
    df_sign, nbr_sig, nbr_levy = signature_extract(
        df_time,
        order=order_sign,
        var_embd=var_embd,
        use_log=use_log,
        use_mat_Levy=use_mat_Levy,
        interpolation_type=interpolation_type,
        var_struct_list=var_struct_seq_list_OG,
        verbose=verbose
    )

    if print_progress:
        print("Signature extraction completed.")

    # ------------------------------------------------------------------
    # 3. Signature preprocessing
    # ------------------------------------------------------------------
    df_sign = preprocess_sign(df_sign, retire_small=False, return_id=False, verbose=False)

    # ------------------------------------------------------------------
    # 4. Prepare Cox-compatible dataset
    # ------------------------------------------------------------------
    df_cox, features_name, id_list = preprocess_cox(
        df_sign,
        debut_etude=var_start,
        return_id=True,
        compute_duree = False
    )

    # ------------------------------------------------------------------
    # 5. Extract survival matrix
    # ------------------------------------------------------------------
    Xt, y, id_list = feat_event_extract(
        df_cox,
        features=features_name,
        var_id=var_id,
        var_DEATH=var_death,
        var_duree=var_duration
    )

    # ------------------------------------------------------------------
    # 6. Construct df_study
    # ------------------------------------------------------------------
    df_study = pd.DataFrame(Xt, columns=features_name)
    df_study.insert(0, var_id, id_list)
    df_study[var_event] = y[var_event].astype(bool)
    df_study[var_time] = y[var_time].astype(float)

    if print_progress:
        print(f"Preprocessing completed in {time.time() - start:.2f} seconds.")
        print(f"Final dataset shape: {df_study.shape}")

    return Xt, y, features_name, nbr_sig, nbr_levy, id_list, df_study