import numpy as np
from itertools import product, combinations
from collections import Counter
from Bio import SeqIO
from io import StringIO
import csv
import math

class RNAFeatureExtractor:
    """
    Complete RNA feature extractor with all methods:
    - Kmer: k-mer nucleotide composition
    - Mismatch: Mismatch k-mer frequency
    - Subsequence: Subsequence profile
    - NAC: Nucleotide composition
    - ENAC: Enhanced nucleotide composition (sliding window)
    - ANF: Accumulated nucleotide frequency
    - NCP: Nucleotide chemical property
    - PSTNPss: Position-specific tri-nucleotide propensity
    - Automatic label assignment based on sequence IDs
    """
    
    RNA_BASES = ['A', 'U', 'C', 'G']
    
    # Chemical properties (Purine, Strong H-bond, H-bond count)
    NCP_PROPERTIES = {
        'A': [1, 1, 1],  # Purine, Strong H-bond, 2 H-bonds
        'U': [0, 0, 1],  # Pyrimidine, Weak H-bond, 2 H-bonds
        'C': [0, 1, 0],  # Pyrimidine, Strong H-bond, 1 H-bond
        'G': [1, 0, 2]   # Purine, Weak H-bond, 3 H-bonds
    }

    @staticmethod
    def assign_label(seq_id):
        """Assign label based on sequence ID (0 for positive, 1 for negative)"""
        seq_id = str(seq_id).lower()  # Convert to lowercase for case-insensitive check
        if 'pos' in seq_id or 'p' in seq_id or '|0|' in seq_id:
            return 0  # Positive class
        return 1  # Negative class

    @staticmethod
    def read_fasta(file_path, include_labels=False):
        """Read RNA sequences from FASTA file"""
        sequences = []
        ids = []
        labels = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq).upper())
            ids.append(record.id)
            if include_labels:
                labels.append(RNAFeatureExtractor.assign_label(record.id))
        if include_labels:
            return ids, sequences, np.array(labels)
        return ids, sequences

    @staticmethod
    def to_csvs(ids, features, feature_names, labels=None):
        """Convert features to CSV string"""
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        # Include labels in header if provided
        if labels is not None:
            writer.writerow(['ID'] + feature_names + ['label'])
            for seq_id, feature_vec, label in zip(ids, features, labels):
                writer.writerow([seq_id] + list(feature_vec) + [label])
        else:
            writer.writerow(['ID'] + feature_names)
            for seq_id, feature_vec in zip(ids, features):
                writer.writerow([seq_id] + list(feature_vec))
                
        return csv_buffer.getvalue()

    # ==================== Kmer ====================
    def kmer(self, sequence, k=3, normalize=True):
        """Calculate k-mer frequencies"""
        kmers = [''.join(p) for p in product(self.RNA_BASES, repeat=k)]
        counts = {kmer: 0 for kmer in kmers}
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer in counts:
                counts[kmer] += 1
        
        if normalize:
            total = max(1, sum(counts.values()))
            return np.array([counts[kmer]/total for kmer in kmers])
        return np.array([counts[kmer] for kmer in kmers])

    def extract_kmer(self, fasta_file, k=3, include_labels=False):
        """Extract k-mer features from FASTA"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        kmers = [''.join(p) for p in product(self.RNA_BASES, repeat=k)]
        features = [self.kmer(seq, k) for seq in sequences]
        feature_names = [f"RNA_{k}mer_{kmer}" for kmer in kmers]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== Mismatch ====================
    def mismatch(self, sequence, k=3, m=1, normalize=True):
        """Calculate mismatch k-mer frequencies"""
        kmers = [''.join(p) for p in product(self.RNA_BASES, repeat=k)]
        counts = {kmer: 0 for kmer in kmers}
        
        for i in range(len(sequence) - k + 1):
            subseq = sequence[i:i+k]
            for kmer in kmers:
                mismatches = sum(1 for a, b in zip(kmer, subseq) if a != b)
                if mismatches <= m:
                    counts[kmer] += 1
        
        if normalize:
            total = max(1, sum(counts.values()))
            return np.array([counts[kmer]/total for kmer in kmers])
        return np.array([counts[kmer] for kmer in kmers])

    def extract_mismatch(self, fasta_file, k=3, m=1, include_labels=False):
        """Extract mismatch features from FASTA"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        kmers = [''.join(p) for p in product(self.RNA_BASES, repeat=k)]
        features = [self.mismatch(seq, k, m) for seq in sequences]
        feature_names = [f"RNA_Mismatch_{k}_{m}_{kmer}" for kmer in kmers]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== Subsequence ====================
    def subsequence(self, sequence, k=3, normalize=True):
        """Calculate subsequence profile"""
        kmers = [''.join(p) for p in product(self.RNA_BASES, repeat=k)]
        counts = {kmer: 0 for kmer in kmers}
        
        for kmer in kmers:
            pos = 0
            while True:
                pos = sequence.find(kmer, pos)
                if pos == -1:
                    break
                counts[kmer] += 1
                pos += 1
        
        if normalize:
            total = max(1, sum(counts.values()))
            return np.array([counts[kmer]/total for kmer in kmers])
        return np.array([counts[kmer] for kmer in kmers])

    def extract_subsequence(self, fasta_file, k=3, include_labels=False):
        """Extract subsequence features from FASTA"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        kmers = [''.join(p) for p in product(self.RNA_BASES, repeat=k)]
        features = [self.subsequence(seq, k) for seq in sequences]
        feature_names = [f"RNA_Subseq_{k}_{kmer}" for kmer in kmers]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== NAC ====================
    def nac(self, sequence, normalize=True):
        """Nucleotide composition"""
        counts = {base: 0 for base in self.RNA_BASES}
        for base in sequence:
            if base in counts:
                counts[base] += 1
        
        if normalize and len(sequence) > 0:
            return np.array([counts[base]/len(sequence) for base in self.RNA_BASES])
        return np.array([counts[base] for base in self.RNA_BASES])

    def extract_nac(self, fasta_file, include_labels=False):
        """Extract nucleotide composition features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.nac(seq) for seq in sequences]
        feature_names = [f"RNA_NAC_{base}" for base in self.RNA_BASES]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== ENAC ====================
    def enac(self, sequence, window=5, normalize=True):
        """Enhanced nucleotide composition (sliding window)"""
        features = []
        for i in range(0, len(sequence) - window + 1):
            window_seq = sequence[i:i+window]
            counts = {base: 0 for base in self.RNA_BASES}
            for base in window_seq:
                if base in counts:
                    counts[base] += 1
            if normalize:
                features.extend([counts[base]/window for base in self.RNA_BASES])
            else:
                features.extend([counts[base] for base in self.RNA_BASES])
        return np.array(features)

    def extract_enac(self, fasta_file, window=5, include_labels=False):
        """Extract ENAC features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        min_len = min(len(seq) for seq in sequences)
        n_windows = max(1, min_len - window + 1)
        features = []
        for seq in sequences:
            feats = self.enac(seq, window)
            features.append(feats[:n_windows*len(self.RNA_BASES)])
        feature_names = [
            f"RNA_ENAC_{base}_w{i}" 
            for i in range(n_windows) 
            for base in self.RNA_BASES
        ]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== ANF ====================
    def anf(self, sequence, L=100):
        """Accumulated nucleotide frequency"""
        features = []
        for j in range(len(sequence)):
            sum_freq = 0
            for i in range(j):
                if sequence[i] == sequence[j]:
                    sum_freq += 1
            features.append(sum_freq / (L if L > 0 else 1))
        return np.array(features)

    def extract_anf(self, fasta_file, L=100, include_labels=False):
        """Extract ANF features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        max_len = max(len(seq) for seq in sequences)
        features = []
        for seq in sequences:
            feats = self.anf(seq, L)
            padded = np.zeros(max_len)
            padded[:len(feats)] = feats
            features.append(padded)
        feature_names = [f"RNA_ANF_pos{i}" for i in range(max_len)]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== NCP ====================
    def ncp(self, sequence):
        """Nucleotide chemical property"""
        features = []
        for base in sequence:
            features.extend(self.NCP_PROPERTIES.get(base, [0, 0, 0]))
        return np.array(features)

    def extract_ncp(self, fasta_file, include_labels=False):
        """Extract NCP features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        max_len = max(len(seq) for seq in sequences)
        features = []
        for seq in sequences:
            feats = self.ncp(seq)
            padded = np.zeros(max_len * 3)
            padded[:len(feats)] = feats
            features.append(padded)
        property_names = ['Purine', 'Strong_Hbond', 'Hbond_Count']
        feature_names = [
            f"RNA_NCP_pos{i//3}_{property_names[i%3]}" 
            for i in range(max_len * 3)
        ]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== PSTNPss ====================
    def pstnpss(self, sequence, positive_samples=None):
        """Position-Specific Tri-Nucleotide Propensity"""
        if positive_samples is None:
            positive_samples = []
            
        # Calculate trinucleotide frequencies in positive samples
        tri_counts = {''.join(tri): 0 for tri in product(self.RNA_BASES, repeat=3)}
        for seq in positive_samples:
            for i in range(len(seq)-2):
                tri = seq[i:i+3]
                if tri in tri_counts:
                    tri_counts[tri] += 1
        total_pos = max(1, sum(tri_counts.values()))
        
        # Calculate trinucleotide frequencies in current sequence
        seq_counts = {''.join(tri): 0 for tri in product(self.RNA_BASES, repeat=3)}
        for i in range(len(sequence)-2):
            tri = sequence[i:i+3]
            if tri in seq_counts:
                seq_counts[tri] += 1
        total_seq = max(1, sum(seq_counts.values()))
        
        # Compute PSTNPss scores
        features = []
        for tri in product(self.RNA_BASES, repeat=3):
            tri_str = ''.join(tri)
            pos_freq = tri_counts[tri_str] / total_pos
            seq_freq = seq_counts[tri_str] / total_seq
            features.append(seq_freq - pos_freq)
        return np.array(features)

    def extract_pstnpss(self, fasta_file, positive_fasta=None, include_labels=False):
        """Extract PSTNPss features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        positive_seqs = self.read_fasta(positive_fasta)[1] if positive_fasta else []
        features = [self.pstnpss(seq, positive_seqs) for seq in sequences]
        feature_names = [
            f"RNA_PSTNPss_{''.join(tri)}" 
            for tri in product(self.RNA_BASES, repeat=3)
        ]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== Unified Interface ====================
    def extract_features(self, fasta_file, methods=['NAC'], params=None, include_labels=False):
        """
        Unified feature extraction interface
        :param fasta_file: Input FASTA file
        :param methods: List of methods to use
        :param params: Dictionary of parameters for specific methods
        :param include_labels: Whether to include automatically assigned labels
        :return: (ids, features, feature_names) or (ids, features, feature_names, labels)
        """
        params = params or {}
        all_features = []
        all_names = []
        ids = None
        labels = None
        
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
        
        if 'Kmer' in methods:
            k = params.get('k', 3)
            features = [self.kmer(seq, k) for seq in sequences]
            feature_names = [f"RNA_{k}mer_{kmer}" for kmer in 
                           [''.join(p) for p in product(self.RNA_BASES, repeat=k)]]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'Mismatch' in methods:
            k = params.get('k', 3)
            m = params.get('m', 1)
            features = [self.mismatch(seq, k, m) for seq in sequences]
            feature_names = [f"RNA_Mismatch_{k}_{m}_{kmer}" for kmer in 
                           [''.join(p) for p in product(self.RNA_BASES, repeat=k)]]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'Subsequence' in methods:
            k = params.get('k', 3)
            features = [self.subsequence(seq, k) for seq in sequences]
            feature_names = [f"RNA_Subseq_{k}_{kmer}" for kmer in 
                           [''.join(p) for p in product(self.RNA_BASES, repeat=k)]]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'NAC' in methods:
            features = [self.nac(seq) for seq in sequences]
            feature_names = [f"RNA_NAC_{base}" for base in self.RNA_BASES]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'ENAC' in methods:
            window = params.get('window', 5)
            min_len = min(len(seq) for seq in sequences)
            n_windows = max(1, min_len - window + 1)
            features = []
            for seq in sequences:
                feats = self.enac(seq, window)
                features.append(feats[:n_windows*len(self.RNA_BASES)])
            feature_names = [f"RNA_ENAC_{base}_w{i}" 
                           for i in range(n_windows) 
                           for base in self.RNA_BASES]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'ANF' in methods:
            L = params.get('L', 100)
            max_len = max(len(seq) for seq in sequences)
            features = []
            for seq in sequences:
                feats = self.anf(seq, L)
                padded = np.zeros(max_len)
                padded[:len(feats)] = feats
                features.append(padded)
            feature_names = [f"RNA_ANF_pos{i}" for i in range(max_len)]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'NCP' in methods:
            max_len = max(len(seq) for seq in sequences)
            features = []
            for seq in sequences:
                feats = self.ncp(seq)
                padded = np.zeros(max_len * 3)
                padded[:len(feats)] = feats
                features.append(padded)
            property_names = ['Purine', 'Strong_Hbond', 'Hbond_Count']
            feature_names = [
                f"RNA_NCP_pos{i//3}_{property_names[i%3]}" 
                for i in range(max_len * 3)
            ]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'PSTNPss' in methods:
            positive_fasta = params.get('positive_fasta')
            positive_seqs = self.read_fasta(positive_fasta)[1] if positive_fasta else []
            features = [self.pstnpss(seq, positive_seqs) for seq in sequences]
            feature_names = [
                f"RNA_PSTNPss_{''.join(tri)}" 
                for tri in product(self.RNA_BASES, repeat=3)
            ]
            all_features.append(features)
            all_names.extend(feature_names)
        
        # Combine all features
        combined_features = np.hstack(all_features) if all_features else np.array([])
        
        if include_labels:
            return ids, combined_features, all_names, labels
        return ids, combined_features, all_names
    
    def to_csv(self, fasta_file, methods=['NAC'], params=None, include_labels=False):
        """Convert extracted features to CSV"""
        if include_labels:
            ids, features, feature_names, labels = self.extract_features(
                fasta_file, methods, params, include_labels=True
            )
            return self.to_csvs(ids, features, feature_names, labels)
        else:
            ids, features, feature_names = self.extract_features(
                fasta_file, methods, params
            )
            return self.to_csvs(ids, features, feature_names)