import numpy as np
import pandas as pd
import os

def synth_indep_given_overlap(df1, df2, 
                              cols1, cols2, 
                              sample_n=100000, save_path=None):
    """
    Generate synthetic samples assuming independence given the overlapping variable.
    
    Args:
        df1 (pd.DataFrame): First marginal dataset with variables cols1.
        df2 (pd.DataFrame): Second marginal dataset with variables cols2.
        cols1 (list of str): Columns in first dataset (includes overlap).
        cols2 (list of str): Columns in second dataset (includes overlap).
        sample_n (int): Number of synthetic samples to generate.
        save_path (str or None): File path to save synthetic dataset CSV. If None, no save.
        
    Returns:
        pd.DataFrame: Synthetic dataset combining variables from both marginals.
    """

    # Identify the overlapping column (assuming exactly one overlap)
    overlap_col = list(set(cols1) & set(cols2))[0]

    # Columns other than the overlap from both datasets
    non_overlap_cols = list((set(cols1) | set(cols2)) - {overlap_col})

    # Calculate proportion distribution of overlap variable from df1 (can also average if desired)
    overlap_prop = df1[overlap_col].value_counts(normalize=True).sort_index()

    # Categories of the overlapping variable (assumes categorical dtype)
    overlap_cats = df1[overlap_col].cat.categories

    # Sample the overlapping variable based on its distribution
    overlap_samples = np.random.choice(a=overlap_cats, size=sample_n, p=overlap_prop)

    # Prepare empty dataframe for results with all desired columns
    synth_df = pd.DataFrame(columns=[overlap_col] + non_overlap_cols)

    # For each category in the overlapping variable, sample from marginals conditioned on that category
    for cat in overlap_cats:
        n_samples_cat = sum(overlap_samples == cat)

        # Sample from df1 without the overlap column
        subset1 = df1[df1[overlap_col] == cat].drop(columns=[overlap_col])
        samples1 = subset1.sample(n=n_samples_cat, replace=True).reset_index(drop=True)

        # Sample from df2 with the overlap column included (to preserve it)
        subset2 = df2[df2[overlap_col] == cat]
        samples2 = subset2.sample(n=n_samples_cat, replace=True).reset_index(drop=True)

        # Combine sampled columns side-by-side
        combined = pd.concat([samples1, samples2], axis=1)

        # Append to the synthetic dataframe
        synth_df = pd.concat([synth_df, combined], ignore_index=True)

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        synth_df.to_csv(save_path, index=False)

    return synth_df


def synth_marginal_preservation(df1, df2, 
                              cols1, cols2, 
                              sample_n=100000, save_path=None):
    
    """
    Return dataframe of synthetic data samples generated using method2.  

    @type  data_name: String e.g. 'compas'
    @param data_name: Name of dataset used to learn joint distribution for synthetic data generation 

    @type  data_folder: Stings e.g. '/data/interim'
    @param data_folder: folder path in relation to cwd 

    @type  data_name: Sting e.g. 'compas_data'
    @param data_name: File name of dataframe

    @type  cols1: list e.g. ['savings', 'occupation']
    @param cols1: list of column names for first marginal dataset

    @type  cols2: list e.g. ['occupation', 'race']
    @param cols2: list of column names for second marginal dataset

    @type  sample_n: int e.g. 100000
    @param sample_n: number of synthetic data samples to generate

    @type  save_folder: Stings e.g. '/data/interim/german_data/method1_emp'
    @param save_folder: folder path in relation to cwd 

    @type  save_name: Sting e.g. 'german_synth_df'
    @param save_name: File name of dataframe

    @rtype:   pd.DataFrame
    @return:  Dataset of synthetic data samples
    """

    
    # obtain label name for overlapping variable 
    overlap_col = list(set(cols1) & set(cols2))[0]

    # get categories for overlapping variable
    overlap_cat = df[overlap_col].cat.categories

    # obtain other column labels
    # Columns other than the overlap from both datasets
    non_overlap_cols = list((set(cols1) | set(cols2)) - {overlap_col})

    # create empty dataset for synthetic dataset
    synth_df = pd.DataFrame(columns=[overlap_col] + non_overlap_cols)

    # sample from first marginal dataset
    joint_samples = df1.sample(n= sample_n, replace = True)
    joint_samples.reset_index(drop = True, inplace=True)

    # iterate over different categories in overlapping variable
    for k in overlap_cat:

        # get samples from joint_samples where overlapping variable = k 
        samples_1 = joint_samples[joint_samples[overlap_col] == k]
        samples_1.reset_index(drop = True, inplace=True)

        # get samples from second marginal dataset conditioned on overlapping variable = k e.g. f(x_3 | x_2 = k)
        gk2 = df2[df2[overlap_col] == k].drop([overlap_col], axis = 1)
        samples_2 = gk2.sample(n= sum(joint_samples[overlap_col] == k), replace = True)
        samples_2.reset_index(drop = True, inplace=True)

        # join samples from marginal datasets 
        synth_df_loop = pd.concat([samples_1, samples_2], axis=1)
        synth_df_loop.reset_index(drop = True, inplace=True)

        # join together datasets for every k in for loop 
        synth_df = pd.concat([synth_df, synth_df_loop])
        synth_df.reset_index(drop = True, inplace=True)

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        synth_df.to_csv(save_path, index=False)

    return synth_df


# ====== Quick test example ======
if __name__ == "__main__":
    # Create dummy data
    np.random.seed(42)
    
    df = pd.DataFrame({
        's': np.random.choice([0, 1], 1000),
        'o': np.random.choice([0, 1], 1000),
        'e': np.random.choice([0, 1], 1000)
    })
    
    for col in df.columns:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')
    
    df1 = df[['s', 'o']].copy()
    df2 = df[['o', 'e']].copy()
    
    def get_joint_distribution(df, cols):
        joint = df.groupby(cols).size().div(len(df))
        return joint

    # Usage

    synth = synth_indep_given_overlap(df1, df2, ['s', 'o'], ['o', 'e'], sample_n=5000)
    print(synth.head())
    print(synth.shape)
    print(get_joint_distribution(synth, ['s', 'o']))
    print(get_joint_distribution(synth, ['o', 'e']))
    synth = synth_marginal_preservation(df1, df2, ['s', 'o'], ['o', 'e'], sample_n=5000)
    print(synth.head())
    print(synth.shape)
    print(get_joint_distribution(synth, ['s', 'o']))
    print(get_joint_distribution(synth, ['o', 'e']))
    
    