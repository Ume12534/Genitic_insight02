import numpy as np
from itertools import product, combinations
from Bio import SeqIO
from io import StringIO
import csv

class DNAFeatureExtractor:
    """
    Unified DNA feature extractor with multiple methods:
    - Kmer: k-mer nucleotide composition
    - RCKmer: Reverse complement k-mer frequency
    - Mismatch: Mismatch k-mer frequency
    - Subsequence: Subsequence profile
    - NAC: Nucleotide composition
    - ANF: Accumulated nucleotide frequency
    - ENAC: Enhanced nucleotide composition (sliding window)
    - Automatic label assignment based on sequence IDs
    """
    
    # DNA nucleotide alphabet and complement
    DNA_BASES = ['A', 'T', 'C', 'G']
    COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    
    @staticmethod
    def assign_label(seq_id):
        """Assign label based on sequence ID (0 for positive, 1 for negative)"""
        seq_id = str(seq_id).lower()  # Convert to lowercase for case-insensitive check
        if 'pos' in seq_id or 'p' in seq_id or '|0|' in seq_id:
            return 0  # Positive class
        return 1  # Negative class

    @staticmethod
    def read_fasta(file_path, include_labels=False):
        """Read DNA sequences from FASTA file"""
        sequences = []
        ids = []
        labels = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq).upper())
            ids.append(record.id)
            if include_labels:
                labels.append(DNAFeatureExtractor.assign_label(record.id))
        if include_labels:
            return ids, sequences, np.array(labels)
        return ids, sequences

    @staticmethod
    def features_to_csv(ids, features, feature_names, labels=None):
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
        """K-mer frequency composition"""
        kmers = [''.join(p) for p in product(self.DNA_BASES, repeat=k)]
        counts = {kmer: 0 for kmer in kmers}
        total = max(1, len(sequence) - k + 1)
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer in counts:
                counts[kmer] += 1
        
        freqs = np.array([counts[kmer] for kmer in kmers])
        return freqs / total if normalize else freqs

    def extract_kmer(self, fasta_file, k=3, include_labels=False):
        """Extract k-mer features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.kmer(seq, k) for seq in sequences]
        kmers = [''.join(p) for p in product(self.DNA_BASES, repeat=k)]
        feature_names = [f"DNA{k}mer_{kmer}" for kmer in kmers]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== RCKmer ====================
    def reverse_complement(self, kmer):
        """Generate reverse complement of DNA k-mer"""
        return ''.join([self.COMPLEMENT[base] for base in reversed(kmer)])

    def rckmer(self, sequence, k=3, normalize=True):
        """Reverse complement k-mer frequency"""
        kmers = set()
        all_kmers = [''.join(p) for p in product(self.DNA_BASES, repeat=k)]
        
        # Get canonical kmers (kmer and its reverse complement)
        for kmer in all_kmers:
            rc = self.reverse_complement(kmer)
            canonical = min(kmer, rc)
            kmers.add(canonical)
        kmers = sorted(kmers)
        
        counts = {kmer: 0 for kmer in kmers}
        total = max(1, len(sequence) - k + 1)
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            rc = self.reverse_complement(kmer)
            canonical = min(kmer, rc)
            if canonical in counts:
                counts[canonical] += 1
        
        freqs = np.array([counts[kmer] for kmer in kmers])
        return freqs / total if normalize else freqs

    def extract_rckmer(self, fasta_file, k=3, include_labels=False):
        """Extract reverse complement k-mer features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.rckmer(seq, k) for seq in sequences]
        
        # Generate canonical kmers for feature names
        kmers = set()
        for kmer in product(self.DNA_BASES, repeat=k):
            canonical = min(''.join(kmer), self.reverse_complement(''.join(kmer)))
            kmers.add(canonical)
        feature_names = [f"DNA_RC{k}mer_{kmer}" for kmer in sorted(kmers)]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== Mismatch ====================
    def mismatch(self, sequence, k=3, m=1, normalize=True):
        """Mismatch k-mer frequency"""
        all_kmers = [''.join(p) for p in product(self.DNA_BASES, repeat=k)]
        counts = {kmer: 0 for kmer in all_kmers}
        total = max(1, len(sequence) - k + 1)
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            for mismatch in self.generate_mismatches(kmer, m):
                if mismatch in counts:
                    counts[mismatch] += 1
        
        freqs = np.array([counts[kmer] for kmer in all_kmers])
        return freqs / total if normalize else freqs

    @staticmethod
    def generate_mismatches(kmer, m):
        """Generate all possible mismatches with up to m substitutions"""
        if m == 0:
            return [kmer]
        mismatches = set()
        for positions in combinations(range(len(kmer)), m):
            for replacements in product(DNAFeatureExtractor.DNA_BASES, repeat=m):
                new_kmer = list(kmer)
                for i, pos in enumerate(positions):
                    new_kmer[pos] = replacements[i]
                mismatches.add(''.join(new_kmer))
        return sorted(mismatches)

    def extract_mismatch(self, fasta_file, k=3, m=1, include_labels=False):
        """Extract mismatch features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.mismatch(seq, k, m) for seq in sequences]
        kmers = [''.join(p) for p in product(self.DNA_BASES, repeat=k)]
        feature_names = [f"DNA{k}mer_m{m}_{kmer}" for kmer in kmers]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== Subsequence ====================
    def subsequence(self, sequence, k=3, normalize=True):
        """Subsequence profile (count of all possible k-length subsequences)"""
        return self.kmer(sequence, k, normalize)  # Same as kmer for DNA

    def extract_subsequence(self, fasta_file, k=3, include_labels=False):
        """Extract subsequence profile features"""
        return self.extract_kmer(fasta_file, k, include_labels)  # Same as kmer for DNA

    # ==================== NAC ====================
    def nac(self, sequence, normalize=True):
        """Nucleotide composition"""
        counts = {base: 0 for base in self.DNA_BASES}
        for base in sequence:
            if base in counts:
                counts[base] += 1
        total = max(1, len(sequence))
        freqs = np.array([counts[base] / total for base in self.DNA_BASES])
        return freqs if normalize else np.array([counts[base] for base in self.DNA_BASES])

    def extract_nac(self, fasta_file, include_labels=False):
        """Extract nucleotide composition features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.nac(seq) for seq in sequences]
        feature_names = [f"DNA_NAC_{base}" for base in self.DNA_BASES]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== ANF ====================
    def anf(self, sequence, L=100):
        """Accumulated nucleotide frequency"""
        counts = {base: np.zeros(L) for base in self.DNA_BASES}
        for i, base in enumerate(sequence):
            if base in counts and i < L:
                counts[base][i] = 1
        
        features = []
        for base in self.DNA_BASES:
            cumsum = np.cumsum(counts[base])
            norm = np.arange(1, L+1)
            features.extend(cumsum / norm)
        return np.array(features)

    def extract_anf(self, fasta_file, L=100, include_labels=False):
        """Extract ANF features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.anf(seq, L) for seq in sequences]
        feature_names = [f"DNA_ANF_{base}_p{i}" 
                        for base in self.DNA_BASES 
                        for i in range(L)]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== ENAC ====================
    def enac(self, sequence, window=5):
        """Enhanced nucleotide composition (sliding window)"""
        features = []
        for i in range(0, len(sequence) - window + 1):
            window_seq = sequence[i:i+window]
            counts = {base: 0 for base in self.DNA_BASES}
            for base in window_seq:
                if base in counts:
                    counts[base] += 1
            features.extend([counts[base]/window for base in self.DNA_BASES])
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
            features.append(feats[:n_windows*len(self.DNA_BASES)])
            
        feature_names = [f"DNA_ENAC_{base}_w{i}" 
                        for i in range(n_windows) 
                        for base in self.DNA_BASES]
        
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
            feature_names = [f"DNA{k}mer_{kmer}" for kmer in 
                           [''.join(p) for p in product(self.DNA_BASES, repeat=k)]]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'RCKmer' in methods:
            k = params.get('k', 3)
            features = [self.rckmer(seq, k) for seq in sequences]
            kmers = set()
            for kmer in product(self.DNA_BASES, repeat=k):
                canonical = min(''.join(kmer), self.reverse_complement(''.join(kmer)))
                kmers.add(canonical)
            feature_names = [f"DNA_RC{k}mer_{kmer}" for kmer in sorted(kmers)]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'Mismatch' in methods:
            k = params.get('k', 3)
            m = params.get('m', 1)
            features = [self.mismatch(seq, k, m) for seq in sequences]
            feature_names = [f"DNA{k}mer_m{m}_{kmer}" for kmer in 
                           [''.join(p) for p in product(self.DNA_BASES, repeat=k)]]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'Subsequence' in methods:
            k = params.get('k', 3)
            features = [self.subsequence(seq, k) for seq in sequences]
            feature_names = [f"DNA_Subseq{k}_{kmer}" for kmer in 
                           [''.join(p) for p in product(self.DNA_BASES, repeat=k)]]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'NAC' in methods:
            features = [self.nac(seq) for seq in sequences]
            feature_names = [f"DNA_NAC_{base}" for base in self.DNA_BASES]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'ANF' in methods:
            L = params.get('L', 100)
            features = [self.anf(seq, L) for seq in sequences]
            feature_names = [f"DNA_ANF_{base}_p{i}" 
                           for base in self.DNA_BASES 
                           for i in range(L)]
            all_features.append(features)
            all_names.extend(feature_names)
        
        if 'ENAC' in methods:
            window = params.get('window', 5)
            min_len = min(len(seq) for seq in sequences)
            n_windows = max(1, min_len - window + 1)
            features = []
            for seq in sequences:
                feats = self.enac(seq, window)
                features.append(feats[:n_windows*len(self.DNA_BASES)])
            feature_names = [f"DNA_ENAC_{base}_w{i}" 
                           for i in range(n_windows) 
                           for base in self.DNA_BASES]
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
            return self.features_to_csv(ids, features, feature_names, labels)
        else:
            ids, features, feature_names = self.extract_features(
                fasta_file, methods, params
            )
            return self.features_to_csv(ids, features, feature_names)