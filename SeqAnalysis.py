import os
import re
import sys
from utils import *
from genetic_code import tables
from itertools import product
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def read_fasta(filepath: str) -> dict:
    """
    Reads a FASTA file and returns a dictionary with headers as keys and sequences as values.

    Parameters:
    filepath (str): The path to the FASTA file.

    Returns:
    dict: A dictionary where the keys are headers and the values are sequences.

    Raises:
    FileNotFoundError: If the file specified by filepath does not exist.
    EmptyFileError: If the file is empty.
    GeneralFASTAError: For any other errors encountered during file reading.
    """
    records = {}
    
    try:
        # Check if the file is empty
        if os.stat(filepath).st_size == 0:
            raise EmptyFileError(filepath)
        
        with open(filepath, mode="r") as fasta_reader:
            
            header = None
            sequence = []
            
            for line in fasta_reader:
                line = line.strip()
                if line.startswith(">"):
                    # Save the previous header and sequence in the dictionary
                    if header:
                        records[header] = ''.join(sequence)
                    # Update header and reset sequence
                    header = line[1:].strip().split()[0]
                    sequence = []
                else:
                    sequence.append(line)
            
            # Save the last header and sequence in the dictionary
            if header:
                records[header] = ''.join(sequence)
                
    except FileNotFoundError as e:
        raise FileNotFoundError(filepath) from e
    except EmptyFileError as e:
        raise e
    except Exception as e:
        raise GeneralFASTAError(filepath, str(e)) from e
    
    # return the result as dictionary
    return records

def read_fastq(filepath: str) -> dict:
    """
    Reads a FASTQ file and returns a dictionary with headers as keys and sequences as values.

    Parameters:
    filepath (str): The path to the FASTQ file.

    Returns:
    dict: A dictionary where the keys are headers and the values are sequences.

    Raises:
    FileNotFoundError: If the file specified by filepath does not exist.
    EmptyFileError: If the file is empty.
    GeneralFASTAError: For any other errors encountered during file reading.
    """
    records = {}
    
    try:
        # Check if the file is empty
        if os.stat(filepath).st_size == 0:
            raise EmptyFileError(filepath)
        
        with open(filepath, mode="r") as fastq_reader:
            while True:
                header = fastq_reader.readline().strip()
                sequence = fastq_reader.readline().strip()
                plus = fastq_reader.readline().strip()
                quality = fastq_reader.readline().strip()

                # If any line is empty, we've reached the end of the file
                if not header or not sequence or not plus or not quality:
                    break

                if not header.startswith("@") or not plus.startswith("+"):
                    raise GeneralFASTAError(filepath, "Invalid FASTQ format")

                header = header[1:].split()[0]  # Remove '@' and get the header
                
                records[header] = sequence

    except FileNotFoundError as e:
        raise FileNotFoundError(filepath) from e
    except EmptyFileError as e:
        raise e
    except Exception as e:
        raise GeneralFASTAError(filepath, str(e)) from e
    
    return records

def GC(sequence: str) -> float:
    """
    Calculate the GC content of a DNA sequence.
    
    Parameters:
    dna_sequence (str): A string representing the DNA sequence.
    
    Returns:
    float: The GC content percentage.
    """
    if sequence.islower():
        sequence = sequence.upper()
    
    if not sequence:
        return 0.0

    gc_count = 0
    for base in sequence:
        if base in ['G', 'C']:
            gc_count += 1

    gc_percentage = round((gc_count / len(sequence)) * 100, 4)
    return gc_percentage

def count_nucleotides(sequence: str, plot: bool = False) -> dict:
    """
    Count the number of each nucleotide in a DNA sequence.
    
    Parameters:
    sequence (str): A string representing the DNA sequence.
    plot (bool): If True, plots a bar graph of the nucleotide counts.
    
    Returns:
    dict: A dictionary with nucleotides as keys and their counts as values.
    """
    if sequence.islower():
        sequence = sequence.upper()
    
    nuc_dict = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for base in sequence:
        nuc_dict[base] = nuc_dict.get(base, 0) + 1

    N = len(sequence)

    if plot:
        ax = plt.bar(nuc_dict.keys(), nuc_dict.values(),
                     color=['tab:blue', 'tab:green', 'tab:orange', 'tab:cyan'])
        for bar, count in zip(ax, nuc_dict.values()):
            height = bar.get_height()
            pcent = round((count / N) * 100, 1)
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{pcent}%', ha='center', va='bottom', color='black')
        plt.xlabel('Nucleotide')
        plt.ylabel('Count')
        plt.title('Nucleotide Count in DNA Sequence')
        plt.show()

    return nuc_dict

def count_dinucleotides(sequence: str, plot: bool = False) -> dict:
    """
    Count the number of each dinucleotide in a DNA sequence.

    Parameters:
    sequence (str): A string representing the DNA sequence.
    plot (bool): If True, plots a bar graph of the dinucleotide counts.

    Returns:
    dict: A dictionary with dinucleotides as keys and their counts as values.
    """
    if sequence.islower():
        sequence = sequence.upper()
    
    dinuc_dict = {''.join(pair): 0 for pair in product('ACGT', repeat=2)}

    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        dinuc_dict[dinuc] = dinuc_dict.get(dinuc, 0) + 1
    
    total = sum(dinuc_dict.values())

    if plot:
        ax = plt.bar(dinuc_dict.keys(), dinuc_dict.values(), color='darkorange')
        for bar, count in zip(ax, dinuc_dict.values()):
            height = bar.get_height()
            pcent = round((count / total) * 100, 1)
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{pcent}%', ha='center', va='bottom', color='black')
        plt.xlabel('Dinucleotide')
        plt.ylabel('Count')
        plt.title('Dinucleotide Count in DNA Sequence')
        plt.xticks(rotation=0)
        plt.show()

    return dinuc_dict

def count_codons(sequence: str, plot: bool = False) -> dict:
    """
    Count the number of each codon in a DNA sequence.

    Parameters:
    sequence (str): A string representing the DNA sequence.
    plot (bool): If True, generates a heatmap of the codon counts.

    Returns:
    dict: A dictionary with codons as keys and their counts as values.
    """
    if sequence.islower():
        sequence = sequence.upper()
    
    codon_dict = {''.join(codon): 0 for codon in product('ACGT', repeat=3)}

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        codon_dict[codon] = codon_dict.get(codon, 0) + 1

    if plot:
        codon_matrix = np.array(list(codon_dict.values())).reshape(8, 8)
        total = np.sum(codon_matrix)
        codon_labels = np.array(list(codon_dict.keys())).reshape(8, 8)
        #print(codon_labels)

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(codon_matrix, cmap="viridis", cbar=True, xticklabels=[], yticklabels=[])

        for i in range(codon_matrix.shape[0]):
            for j in range(codon_matrix.shape[1]):
                pcent = round((codon_matrix[i, j] / total) * 100, 1)
                ax.text(j + 0.5, i + 0.5, f'{codon_labels[i, j]}\n{pcent}%',
                        ha='center', va='center', color='white')

        plt.title('Codon Count Heatmap')
        plt.show()

    return codon_dict

def complement(dna: str) -> str:
    """
    Generate the complement of a DNA sequence.

    Parameters:
    dna (str): A string representing the DNA sequence.

    Returns:
    str: The complementary DNA sequence.
    """
    # Convert to uppercase if the input is in lowercase
    if dna.islower():
        dna = dna.upper()
    
    # Dictionary for complement bases
    complement_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    
    # Generate complement sequence using list comprehension
    complement_dna = "".join([complement_dict.get(base, 'N') for base in dna])
    return complement_dna

def reverse_complement(dna: str) -> str:
    """
    Generate the reverse complement of a DNA sequence.

    Parameters:
    dna (str): A string representing the DNA sequence.

    Returns:
    str: The reverse complementary DNA sequence.
    """
    # Get the complement and then reverse the sequence
    rev_complement_dna = complement(dna)[::-1]
    return rev_complement_dna

def transcribe(dna: str) -> str:
    """
    Transcribe a DNA sequence into RNA.

    Parameters:
    dna (str): A string representing the DNA sequence.

    Returns:
    str: The transcribed RNA sequence.
    """
    # Convert to uppercase if the input is in lowercase
    if dna.islower():
        dna = dna.upper()
    
    # Replace all thymine (T) with uracil (U)
    rna = dna.replace("T", "U")
    return rna

def translate(dna: str, table: int = 1) -> str:
    """
    Translates a DNA sequence into a protein sequence using the specified codon table.

    Args:
        dna (str): The DNA sequence to be translated. It is expected to be a string of nucleotide bases (A, T, C, G).
        table (int, optional): The index of the codon table to use for translation. Defaults to 1.

    Returns:
        str: The translated protein sequence. Each codon (three bases) in the DNA sequence is converted to the corresponding amino acid.

    Raises:
        ValueError: If the DNA sequence contains invalid characters or if the length of the DNA sequence is not a multiple of 3.
    """
    # Convert the DNA sequence to uppercase if it is in lowercase
    if dna.islower():
        dna = dna.upper()
    # Get sequence length
    N = len(dna)
    
    # Validate DNA sequence
    valid_bases = {'A', 'T', 'C', 'G'}
    if not set(dna).issubset(valid_bases):
        raise ValueError("Invalid DNA sequence: Contains characters other than A, T, C, G")
    if N % 3 != 0:
        print("Warning: DNA sequence length is not a multiple of 3. The incomplete trailing bases will be ignored.")

    # Retrieve the codon table from the provided tables dictionary
    codon_table = tables[int(table)]['table']

    # Translate DNA sequence to protein sequence
    protein = "".join([codon_table[dna[i:i+3]] for i in range(0, N - (N%3), 3)])

    return protein

def translate_in_six_frames(dna: str, table: int = 1) -> dict:
    """
    Translates a DNA sequence in all six reading frames.

    Args:
        dna (str): The DNA sequence to be translated.
        table (int, optional): The index of the codon table to use for translation. Defaults to 1.

    Returns:
        dict: A dictionary with six translated protein sequences corresponding to the six reading frames.
    """
    # Get the reverse complement of the DNA sequence
    rev_dna = reverse_complement(dna)

    # Perform six-frame translation
    frames = {}
    for frame in range(3):
        frames[f"{frame + 1}"] = translate(dna[frame:], table)
        frames[f"-{frame + 1}"] = translate(rev_dna[frame:], table)
    
    return frames

def hamming_distance(seqA: str, seqB: str) -> int:
    """
    Calculate the Hamming distance between two DNA sequences.

    Parameters:
    seqA (str): The first DNA sequence.
    seqB (str): The second DNA sequence.

    Returns:
    int: The Hamming distance between the two sequences.
    """
    # Ensure both sequences are of equal length
    assert len(seqA) == len(seqB), "The DNA sequences must be of equal length."

    # Count the number of differing bases
    distance = sum([1 for base1, base2 in zip(seqA, seqB) if base1 != base2])
    return distance

def transition_transversion_ratio(seqA: str, seqB: str) -> float:
    """
    Calculate the transition/transversion ratio between two DNA sequences.
    The transition/transversion ratio between homologous strands of DNA is generally about 2.

    Parameters:
    seqA (str): The first DNA sequence.
    seqB (str): The second DNA sequence.

    Returns:
    float: The transition/transversion ratio.
    """
    assert len(seqA) == len(seqB), "The DNA sequences must be of equal length."

    transitions, transversions = 0, 0
    transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

    for base1, base2 in zip(seqA, seqB):
        if base1 != base2:
            if (base1, base2) in transition_pairs:
                transitions += 1
            else:
                transversions += 1

    if transversions == 0:
        return float('inf') if transitions > 0 else float('nan')

    ratio = transitions / transversions
    return ratio

if __name__ == "__main__":
    #records = read_fasta("data/sequences.fasta")
    #sequence = records[list(records.keys())[0]]
    #print(sequence)
    #records = read_fasta("data/empty_seq.txt")

    #records = read_fastq("data/sequences.fastq")
    #print(records)
    #records = read_fastq("data/empty_seq.txt")

    #print(GC(sequence))
    #print(count_nucleotides(sequence, plot=True))
    #print(count_dinucleotides(sequence, plot=True))
    #print(count_codons(sequence, plot=True))

    seqA = "GCAACGCACAACGAAAACCCTTAGGGACTGGATTATTTCGTGATCGTTGTAGTTATTGGAAGTACGGGCATCAACCCAGTT"
    seqB = "TTATCTGACAAAGAAAGCCGTCAACGGCTGGATAATTTCGCGATCGTGCTGGTTACTGGCGGTACGAGTGTTCCTTTGGGT"
    seqC = "TTATCTGACAAAGAAAGCCGTCAACNNNNGGATAATTTCGCGATCGTGCTGGTTACTGGCGGTACGAGTGTTCCTTTGGGT"

    print("Complement:", complement(seqC))
    print("Reverse Complement:", reverse_complement(seqC))
    print("Transcribed RNA:", transcribe(seqC))
    print("Hamming Distance:", hamming_distance(seqA, seqB))
    print("Transition/Transversion ratio:", transition_transversion_ratio(seqA, seqB))
    print(translate(seqA))
    print(translate_in_six_frames(seqA))
    print(translate(seqC))
