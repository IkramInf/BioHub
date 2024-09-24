from typing import Tuple, List, Dict, Optional
import numpy as np

def create_substitution_matrix() -> Dict[Tuple[str, str], int]:
    """
    Create a substitution matrix for DNA sequence alignment.

    This function generates a substitution matrix that provides scores for aligning 
    different nucleotide bases (A, T, G, C). The scoring is as follows:
        - A match between the same bases (e.g., A-A, T-T) is given a score of 5.
        - A mismatch between complementary bases (A-T, T-A, G-C, C-G) is penalized with a score of -4.
        - Any other mismatch (e.g., A-G, T-C) is penalized with a score of -3.

    Returns:
        Dict[Tuple[str, str], int]: A dictionary where keys are tuples of base pairs and values are their corresponding scores.
    """
    bases = ['A', 'T', 'G', 'C']
    substitution_matrix = {(b1, b2): 5 if b1 == b2 else -4 if (b1, b2) in [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')] else -3
              for b1 in bases for b2 in bases}
    return substitution_matrix

def find_max_local(matrix: np.ndarray) -> Tuple[int, int, int]:
    """
    Find the maximum value in a matrix and its position.

    This function takes a NumPy array (matrix) as input and identifies the maximum value in the matrix.
    It also returns the row and column indices where this maximum value is located.

    Args:
        matrix (np.ndarray): A 2D NumPy array representing the matrix.

    Returns:
        Tuple[int, int, int]: A tuple containing:
            - score (int): The maximum value in the matrix.
            - i (int): The row index of the maximum value.
            - j (int): The column index of the maximum value.
    """
    score = np.max(matrix)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return score, i, j

def dynamic_programming(A: str, B: str, match_score: int=2, mismatch_score: int=-1, subM: Dict[Tuple[str, str], int]=None, 
                        gap_score: int = -1, strategy: str = "global") -> np.ndarray:
    """
    Perform global or local sequence alignment using dynamic programming.

    This function implements a dynamic programming approach to compute a score matrix for the alignment
    of two sequences A and B. It supports both global (Needleman-Wunsch) and local (Smith-Waterman) alignment
    strategies, with customizable match, mismatch, and gap scores. Optionally, a substitution matrix can be 
    provided.

    Args:
        A (str): The first sequence to align.
        B (str): The second sequence to align.
        match_score (int, optional): The score for a matching pair of bases (default is 2).
        mismatch_score (int, optional): The penalty score for a mismatching pair of bases (default is -1).
        subM (Dict[Tuple[str, str], int], optional): A substitution matrix as a dictionary with base pair tuples as keys 
                                                     and their corresponding score as values (default is None, in which case 
                                                     match/mismatch scoring is used).
        gap_score (int, optional): The penalty score for introducing a gap in the alignment (default is -1).
        strategy (str, optional): The alignment strategy, either "global" for global alignment (Needleman-Wunsch) or 
                                  "local" for local alignment (Smith-Waterman) (default is "global").

    Returns:
        np.ndarray: A score matrix representing the alignment scores for the two sequences.

    Raises:
        ValueError: If an unsupported strategy is provided.
    """
    M, N = len(A) + 1, len(B) + 1
    scoreM = np.zeros((M, N), dtype=int)
    
    if strategy == "global":
        scoreM[0] = np.arange(N) * gap_score
        scoreM[:, 0] = np.arange(M) * gap_score
    
    for i in range(1, M):
        for j in range(1, N):
            score = subM[(A[i-1], B[j-1])] if subM is not None else (match_score if A[i-1] == B[j-1] else mismatch_score)
            
            if strategy == "local":
                scoreM[i, j] = max(0, scoreM[i-1, j-1] + score, scoreM[i, j-1] + gap_score, scoreM[i-1, j] + gap_score)
            else:
                scoreM[i, j] = max(scoreM[i-1, j-1] + score, scoreM[i, j-1] + gap_score, scoreM[i-1, j] + gap_score)
    
    return scoreM

def backtracking(A: str, B: str, scoreM: np.ndarray, match_score: int=2, mismatch_score: int=-1,
        subM: Dict[Tuple[str, str], int]=None, gap_score: int = -1, strategy: str = "global") -> Tuple[str, str, int]:
    """
    Perform backtracking to recover the optimal sequence alignment based on the score matrix.

    This function traces back through the score matrix generated from a dynamic programming algorithm to 
    construct the aligned sequences. It supports both global and local alignment strategies.

    Args:
        A (str): The first sequence to align.
        B (str): The second sequence to align.
        scoreM (np.ndarray): The score matrix obtained from the dynamic programming step.
        match_score (int, optional): The score for a matching pair of bases (default is 2).
        mismatch_score (int, optional): The penalty score for a mismatching pair of bases (default is -1).
        subM (Dict[Tuple[str, str], int], optional): A substitution matrix as a dictionary with base pair tuples as keys 
                                                     and their corresponding score as values (default is None, in which case 
                                                     match/mismatch scoring is used).
        gap_score (int, optional): The penalty score for introducing a gap in the alignment (default is -1).
        strategy (str, optional): The alignment strategy, either "global" for global alignment (Needleman-Wunsch) or 
                                  "local" for local alignment (Smith-Waterman) (default is "global").

    Returns:
        Tuple[str, str, int]: A tuple containing:
            - alignmentA (str): The aligned version of the first sequence A.
            - alignmentB (str): The aligned version of the second sequence B.
            - align_score (int): The score of the alignment.

    Raises:
        ValueError: If an unsupported alignment strategy is provided.
    """
    alignmentA, alignmentB = [], []
    m, n = len(A), len(B)
    
    if strategy == "local":
        align_score, m, n = find_max_local(scoreM)
    else:
        align_score = scoreM[m, n]
    
    while m > 0 and n > 0:
        score = scoreM[m, n]
        if strategy == "local" and score == 0:
            break
        
        sub_score = subM[(A[m-1], B[n-1])] if subM is not None else (match_score if A[m-1] == B[n-1] else mismatch_score)
        
        if score == scoreM[m-1, n-1] + sub_score:
            alignmentA.append(A[m-1])
            alignmentB.append(B[n-1])
            m -= 1
            n -= 1
        elif score == scoreM[m-1, n] + gap_score:
            alignmentA.append(A[m-1])
            alignmentB.append('-')
            m -= 1
        else:
            alignmentA.append('-')
            alignmentB.append(B[n-1])
            n -= 1
    
    if strategy == "global":
        alignmentA.extend(A[m-1::-1])
        alignmentB.extend('-' * m)
        alignmentA.extend('-' * n)
        alignmentB.extend(B[n-1::-1])
    
    return (''.join(alignmentA[::-1]), ''.join(alignmentB[::-1]), align_score)

def align(seq1: str, seq2: str, match_score: int=2, mismatch_score: int=-1, substitution_matrix: Dict[Tuple[str, str], int]=None,
          gap_score: int = -1, strategy: str = "global") -> Tuple[Tuple[str, str, int], np.ndarray]:
    """
    Perform sequence alignment between two sequences using dynamic programming and backtracking.

    This function first computes the alignment score matrix using dynamic programming and then 
    uses backtracking to recover the optimal alignment. It supports both global (Needleman-Wunsch) 
    and local (Smith-Waterman) alignment strategies.

    Args:
        seq1 (str): The first sequence to align.
        seq2 (str): The second sequence to align.
        match_score (int, optional): The score for matching characters in the sequences (default is 2).
        mismatch_score (int, optional): The penalty score for mismatching characters (default is -1).
        substitution_matrix (Dict[Tuple[str, str], int], optional): A substitution matrix in the form of a dictionary with 
                                                                   tuples of characters as keys and scores as values. If None, 
                                                                   the default match/mismatch scoring is used.
        gap_score (int, optional): The penalty for introducing a gap in the alignment (default is -1).
        strategy (str, optional): The alignment strategy to use. "global" for global alignment (Needleman-Wunsch) or 
                                  "local" for local alignment (Smith-Waterman) (default is "global").

    Returns:
        Tuple[Tuple[str, str, int], np.ndarray]: A tuple containing:
            - alignment (Tuple[str, str, int]): The aligned sequences and the alignment score.
            - score_matrix (np.ndarray): The score matrix generated by the dynamic programming step.
    """
    score_matrix = dynamic_programming(seq1, seq2, match_score=match_score, mismatch_score=mismatch_score,
                                subM=substitution_matrix, gap_score=gap_score, strategy=strategy)
    alignment = backtracking(seq1, seq2, score_matrix, match_score=match_score, mismatch_score=mismatch_score,
                        subM=substitution_matrix, gap_score=gap_score, strategy=strategy)
    return (alignment, score_matrix)


def print_score_matrix(s1: str, s2: str, mat: np.ndarray) -> None:
    """
    Pretty print function for a score matrix using a NumPy array.

    Args:
        s1 (str): The first sequence (usually representing rows).
        s2 (str): The second sequence (usually representing columns).
        mat (np.ndarray): The score matrix to be printed, represented as a NumPy array.

    Raises:
        TypeError: If the input matrix `mat` is not a NumPy ndarray.
    """
    
    # Error handling to ensure mat is a NumPy ndarray
    if not isinstance(mat, np.ndarray):
        raise TypeError("The matrix 'mat' must be a NumPy ndarray.")
    
    # Prepend filler characters to s1 and s2 for alignment
    s1 = '-' + s1
    s2 = ' -' + s2
    
    # Print the header row (s2)
    print(''.join(['%5s' % aa for aa in s2]))  # Convert s2 to a list of 5-character strings, then join into a single string
    
    # Iterate through each row in the NumPy array
    for i in range(mat.shape[0]):
        vals = ['%5i' % val for val in mat[i]]  # Convert this row's scores to a list of formatted strings
        vals.insert(0, '%5s' % s1[i])           # Add the corresponding character from s1 to the front
        print(''.join(vals))                    # Join the list elements into a single string and print

def print_alignment(a: Tuple[str, str, int]) -> None:
    """
    Prints the aligned sequences with a visual representation of matches.

    Given an alignment tuple containing two aligned sequences and the alignment score,
    this function prints the sequences, showing matching positions with a '|' character
    and non-matching positions with a space. The alignment score is also displayed at the end.

    Args:
        a (Tuple[str, str, int]): A tuple where:
            - The first element is the first aligned sequence.
            - The second element is the second aligned sequence.
            - The third element is the alignment score.

    Returns:
        None: This function prints the alignment to the console.
    """
    seq1, seq2, score = a
    match = ''.join('|' if s1 == s2 else ' ' for s1, s2 in zip(seq1, seq2))
    print('\n'.join([seq1, match, seq2, f'', f'Score = {score}']))

def pairwise_alignment(seq1: str, seq2: str, match_score: int=2, mismatch_score: int=-1,
                       substitution_matrix: Dict[Tuple[str, str], int]=None, gap_score: int = -1, strat: str = "global") -> None:
    """
    Perform pairwise sequence alignment and print the score matrix and alignment.

    This function aligns two sequences using dynamic programming based on the provided scoring scheme 
    (match, mismatch, gap scores, and optionally a substitution matrix). It supports both global and local 
    alignment strategies. After computing the alignment, it prints the score matrix and the aligned sequences 
    with their alignment score.

    Args:
        seq1 (str): The first sequence to align.
        seq2 (str): The second sequence to align.
        match_score (int, optional): The score for matching characters in the sequences (default is 2).
        mismatch_score (int, optional): The penalty score for mismatching characters (default is -1).
        substitution_matrix (Dict[Tuple[str, str], int], optional): A dictionary representing a substitution matrix, where
                                                                   keys are character pairs and values are scores. If None, 
                                                                   the default match/mismatch scoring is used (default is None).
        gap_score (int, optional): The penalty for introducing a gap in the alignment (default is -1).
        strat (str, optional): The alignment strategy, either "global" for Needleman-Wunsch or "local" for Smith-Waterman 
                               (default is "global").

    Returns:
        None: This function prints the alignment score matrix and the aligned sequences with the alignment score.
    """
    alignment, score_matrix = align(seq1, seq2, match_score=match_score, mismatch_score=mismatch_score,
                                    substitution_matrix=substitution_matrix, gap_score=gap_score, strategy=strat)
    print_score_matrix(seq1, seq2, score_matrix)
    print()
    print_alignment(alignment)

if __name__ == "__main__":
    seq1 = "ACGTACGT"
    seq2 = "ACGTACGTACGT"

    # Sample Call 1
    pairwise_alignment(seq1, seq2, match_score=2, mismatch_score=-1, gap_score=-2, strat="local")
    
    # Sample Call 2
    substitution_matrix = create_substitution_matrix()
    pairwise_alignment(seq1, seq2, substitution_matrix=substitution_matrix, gap_score=-2, strat="global")
    
    # Sample Call 3
    # Supports Biopython Substitution matrix
    from Bio.Align import substitution_matrices
    submat = substitution_matrices.load('NUC.4.4')
    pairwise_alignment(seq1, seq2, substitution_matrix=submat, gap_score=-2, strat="local")


    
