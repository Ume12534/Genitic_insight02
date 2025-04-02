import numpy as np
from Bio import SeqIO
from collections import Counter

class SequenceFeatureExtractor:
    def __init__(self, fasta_file):
        """
        Initialize the feature extractor with a FASTA file.

        Args:
            fasta_file (str): Path to the FASTA file containing sequences.
        """
        self.fasta_file = fasta_file
        self.sequences = self._load_sequences()
        self.sequence_type = self._detect_sequence_type()

    def _load_sequences(self):
        """
        Load and validate sequences from the FASTA file.

        Returns:
            list: A list of tuples containing (sequence_id, sequence).
        """
        sequences = []
        for record in SeqIO.parse(self.fasta_file, "fasta"):
            seq = str(record.seq).upper()
            sequences.append((record.id, seq))
        return sequences

    def _detect_sequence_type(self):
        """
        Detect whether the sequences are DNA or protein.

        Returns:
            str: "DNA" or "Protein".
        """
        valid_dna = set("ACGT")
        valid_protein = set("ACDEFGHIKLMNPQRSTVWY")

        for _, seq in self.sequences:
            if all(nucleotide in valid_dna for nucleotide in seq):
                return "DNA"
            elif all(aa in valid_protein for aa in seq):
                return "Protein"
            else:
                raise ValueError("Invalid sequence: contains non-standard characters.")
        return "Unknown"

    def aac(self):
        """
        Calculate Amino Acid Composition (AAC) for protein sequences.

        Returns:
            np.ndarray: A 2D array where each row represents the AAC of a sequence.
        """
        if self.sequence_type != "Protein":
            raise ValueError("AAC is only applicable to protein sequences.")

        features = []
        for _, seq in self.sequences:
            counts = Counter(seq)
            total = len(seq)
            aac = [counts.get(aa, 0) / total for aa in "ACDEFGHIKLMNPQRSTVWY"]
            features.append(aac)
        return np.array(features)

    def paac(self, lambdaValue=3, weight=0.05):
        """
        Calculate Pseudo Amino Acid Composition (PAAC) for protein sequences.

        Args:
            lambdaValue (int): Number of sequence-order correlation factors.
            weight (float): Weight factor for sequence-order features.

        Returns:
            np.ndarray: A 2D array where each row represents the PAAC of a sequence.
        """
        if self.sequence_type != "Protein":
            raise ValueError("PAAC is only applicable to protein sequences.")

        features = []
        aac_features = self.aac()

        # Hydrophobicity values for standard amino acids
        hydrophobicity = {
            'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74,
            'F': 1.19, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'K': -1.50, 'L': 1.06, 'M': 0.64, 'N': -0.78,
            'P': 0.12, 'Q': -0.85, 'R': -2.53, 'S': -0.18,
            'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
        }

        for (_, seq), aac in zip(self.sequences, aac_features):
            theta = []
            for lag in range(1, lambdaValue + 1):
                total = 0
                for j in range(len(seq) - lag):
                    aa1 = seq[j]
                    aa2 = seq[j + lag]
                    total += (hydrophobicity[aa1] - hydrophobicity[aa2]) ** 2
                theta.append(total / (len(seq) - lag))

            # Combine AAC and PAAC features
            denominator = 1 + weight * sum(theta)
            paac = [(weight * t) / denominator for t in theta]
            full_features = list(aac) + paac
            features.append(full_features)

        return np.array(features)

    def gc_content(self):
        """
        Calculate GC Content for DNA sequences.

        Returns:
            np.ndarray: A 1D array where each value represents the GC content of a sequence.
        """
        if self.sequence_type != "DNA":
            raise ValueError("GC Content is only applicable to DNA sequences.")

        features = []
        for _, seq in self.sequences:
            gc_count = seq.count("G") + seq.count("C")
            gc_content = gc_count / len(seq)
            features.append([gc_content])
        return np.array(features)

    def kmer_frequency(self, k=3):
        """
        Calculate K-mer Frequency for DNA sequences.

        Args:
            k (int): Length of the K-mer.

        Returns:
            np.ndarray: A 2D array where each row represents the K-mer frequencies of a sequence.
        """
        if self.sequence_type != "DNA":
            raise ValueError("K-mer Frequency is only applicable to DNA sequences.")

        features = []
        for _, seq in self.sequences:
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            counts = Counter(kmers)
            total = len(kmers)
            kmer_freq = [counts.get(kmer, 0) / total for kmer in sorted(counts)]
            features.append(kmer_freq)
        return np.array(features)

    def extract_features(self, descriptor_type, params=None):
        """
        Main method to extract features based on the descriptor type.

        Args:
            descriptor_type (str): Type of descriptor (e.g., 'AAC', 'PAAC', 'GC', 'Kmer').
            params (dict): Parameters for the descriptor (e.g., lambdaValue, weight, k).

        Returns:
            np.ndarray: A 2D array of extracted features.
        """
        if params is None:
            params = {}

        if descriptor_type == "AAC":
            return self.aac()
        elif descriptor_type == "PAAC":
            return self.paac(**params)
        elif descriptor_type == "GC":
            return self.gc_content()
        elif descriptor_type == "Kmer":
            return self.kmer_frequency(**params)
        else:
            raise ValueError(f"Invalid descriptor type: {descriptor_type}")