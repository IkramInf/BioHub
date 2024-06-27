import os
import re
import sys
from utils import *

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


if __name__ == "__main__":
    records = read_fasta("data/sequence.fasta")
    print(records)
    records = read_fasta("data/empty_seq.fasta")