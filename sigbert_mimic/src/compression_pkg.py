import numpy as np
import torch

# PCA
def pca_compression(
    df_data,
    bar_p,
    var_embd='embeddings',
    verbose=False
):
    """
    Computes a linear compression of high-dimensional embeddings using PCA.

    Given a DataFrame containing p-dimensional embedding vectors, this function 
    returns their projection onto a lower-dimensional subspace of dimension bar_p 
    using the top principal components. The projection matrix R is computed via 
    truncated singular value decomposition (SVD).

    Parameters:
    - df_data: DataFrame containing a column with embedding vectors.
    - bar_p: Size of compression shape.
    - var_embd: Name of the column with embeddings (default: 'embeddings').
    - verbose: If True, prints explained variance ratio and dimension used.

    Returns:
    - V_proj: Compressed embeddings of shape (N, bar_p). Original size of df: (N, p)
    - R_opt: Compression matrix of shape (bar_p, p) derived from PCA.
    """

    df = df_data.copy()

    embeddings = torch.tensor(np.stack(df[var_embd].values), dtype=torch.float32)
    V_numpy = embeddings.numpy()

    """
    for np.stack, the subjects stack up horizontally, from top to the bottom instead of from left to right. 
    ie. each row represents the individual observant; each column represents one of the interested features.

    df[var_embd]: Series
    .values: Convert the Series into a NumPy Object Array (an array containing arrays).

    V_numpy: Although embeddings(var_embd) appear to be numbers, they are often stored as lists or float64 in a DataFrame.
             Thus, by .numpy(), converting PyTorch Tensor back to NumPy Array so as to conduct the further process.
    """

    V_centered = V_numpy - V_numpy.mean(axis=0)
    V = torch.tensor(V_centered, dtype=torch.float32)
    N, p = V.shape

    """
    V: Converting V_centered from NumPy back to a PyTorch Tensor, which prepares it for subsequent matrix operations (SVD or eigenvalue decomposition).
    """

    # SVD
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    R_opt = Vh[:bar_p].numpy()

    """
    torch.linalg.svd returns three tuple(元組)and distributes respectedly to U, S, Vh, which are defined manually.
    U, S Vh: decomposed V by SVD. Recall: A = UD(V^T) (Deep Learning by Ian Goodfellow, page 59, Chapter 2, formula(2.43)
    
    U: Left Singular Vectors
    S: Singular Values, one dimensional. 
       Though in Deep Learning by Ian Goodfellow, Σ = S is defined as a diagonal yet unnecessarily square matrix whose size is m*n,
       linalg.svd() stores only the diagonal components and converts it into an one-dim vector. 
       If want to use exactly the same as theory, one should manually adjust the codes.
    Vh: Right Singular Vectors (conjugate transpose matrix). This is the origin of the PCA projection matrix.

    R_opt: take the columns from the first to the bar_p and convert it into Numpy-used
    """

    # Explained variance
    total_variance = (S**2).sum().item()
    explained_variance = (S[:bar_p]**2).sum().item()
    explained_ratio = explained_variance / total_variance

    """
   .item(): Transform a Tensor with only one element into a standard Python float.
            Note that it can only be used on Tensors that contain only ONE element. 
            If the Tensor contains more than two numbers, calling it will result in an error.
    """

    if verbose:
        print(f"Compression dimension (bar_p): {bar_p}")
        print(f"Explained variance ratio: {explained_ratio:.4%}")

    return V_numpy @ R_opt.T, R_opt

    """
    when verbose = TRUE, run the if...; when = FALSE (as default), don't run

    V_numpy @ R_opt.T: Matrix multiplication (@), project the original data into a new low-dimensional space.
    R_opt = Vh[:bar_p].numpy() as definition
    """


def apply_linear_projection(df_input, R, var_embd='embeddings'):
    """
    Applies a fixed linear projection matrix to compress high-dimensional embeddings
    into a lower-dimensional space.

    This is typically used at test time, where a projection matrix R
    (obtained during training) is used to map input embeddings to a reduced space.
    The method performs a linear transformation of the form:
        Z_projected = R @ Z_original.T

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame containing the input embeddings in column `var_embd`.
    R : np.ndarray
        .ndarray(): n-dimensional array. Not a list, not an one dimensional vector.
        The linear projection matrix (shape [k, p]) used to compress the embeddings
        from original dimension p to compressed dimension k.
    var_embd : str, default='embeddings'
        Name of the column in `df_input` containing the original embedding vectors (as lists or arrays).

    Returns
    -------
    df_projected : pd.DataFrame
        Copy of the original DataFrame with compressed embeddings in `var_embd`.
    """

    df_projected = df_input.copy()

    # Stack embeddings into a matrix shape [n_samples, p]
    Z_original = np.vstack(df_projected[var_embd].values)

    # Apply the linear projection: shape [n_samples, k]
    Z_compressed = (R @ Z_original.T).T

    # Convert each row back to a list to store in the DataFrame
    Z_compressed_list = [list(row) for row in Z_compressed]

    """
    Convert the results of NumPy matrix operations back to the format commonly used for storing Pandas DataFrames.
    To avoid errors caused by Pandas attempting to split the matrix(Z_compressed) to fill different fields, or by mismatched shapes.
    """

    # Replace the embedding column with the compressed version
    df_projected[var_embd] = Z_compressed_list

    return df_projected