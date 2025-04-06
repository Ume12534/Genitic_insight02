import numpy as np
from itertools import product
from collections import Counter
import math
from Bio import SeqIO
from io import StringIO
import csv

class ProteinFeatureExtractor:
    """
    Unified protein feature extractor with multiple methods:
    - AAC: Amino Acid Composition
    - PAAC: Pseudo Amino Acid Composition
    - EAAC: Enhanced Amino Acid Composition (sliding window)
    - CKSAAP: Composition of k-spaced Amino Acid Pairs
    - DPC: Dipeptide Composition
    - DDE: Dipeptide Deviation from Expected mean
    - TPC: Tripeptide Composition
    - Automatic label assignment based on sequence IDs
    """
    
    # Standard 20 amino acids
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Hydrophobicity values (Kyte-Doolittle scale)
    HYDROPHOBICITY = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
        'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
        'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
        'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
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
        """Read protein sequences from FASTA file"""
        sequences = []
        ids = []
        labels = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq).upper())
            ids.append(record.id)
            if include_labels:
                labels.append(ProteinFeatureExtractor.assign_label(record.id))
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

    # ==================== AAC ====================
    def aac(self, sequence, normalize=True):
        """Amino Acid Composition"""
        counts = {aa: 0 for aa in self.AMINO_ACIDS}
        for aa in sequence:
            if aa in counts:
                counts[aa] += 1
        if normalize and len(sequence) > 0:
            return np.array([counts[aa]/len(sequence) for aa in self.AMINO_ACIDS])
        return np.array([counts[aa] for aa in self.AMINO_ACIDS])
    
    def extract_aac(self, fasta_file, normalize=True, include_labels=False):
        """Extract AAC features from FASTA"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.aac(seq, normalize) for seq in sequences]
        feature_names = [f"AAC_{aa}" for aa in self.AMINO_ACIDS]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== PAAC ====================
    def paac(self, sequence, lambda_val=5, w=0.05):
        """Pseudo Amino Acid Composition"""
        # Standard AAC part
        aac_features = self.aac(sequence)
        
        # Sequence order correlation part
        hydro = [self.HYDROPHOBICITY[aa] for aa in sequence if aa in self.HYDROPHOBICITY]
        if len(hydro) < 2:
            return np.concatenate([aac_features, np.zeros(lambda_val)])
        
        theta = []
        for j in range(1, lambda_val + 1):
            sum_val = 0.0
            for i in range(len(hydro) - j):
                sum_val += (hydro[i] - hydro[i+j]) ** 2
            theta.append(sum_val / (len(hydro) - j))
        
        # Combine features
        paac_features = np.concatenate([
            aac_features * (1 - w),
            (w * np.array(theta)) / (1 + w * sum(theta))
        ])
        return paac_features
    
    def extract_paac(self, fasta_file, lambda_val=5, w=0.05, include_labels=False):
        """Extract PAAC features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        features = [self.paac(seq, lambda_val, w) for seq in sequences]
        feature_names = (
            [f"PAAC_AAC_{aa}" for aa in self.AMINO_ACIDS] +
            [f"PAAC_theta_{i}" for i in range(1, lambda_val + 1)]
        )
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== EAAC ====================
    def eaac(self, sequence, window=5, normalize=True):
        """Enhanced Amino Acid Composition (sliding window)"""
        features = []
        for i in range(0, len(sequence) - window + 1):
            window_seq = sequence[i:i+window]
            counts = {aa: 0 for aa in self.AMINO_ACIDS}
            for aa in window_seq:
                if aa in counts:
                    counts[aa] += 1
            if normalize:
                features.extend([counts[aa]/window for aa in self.AMINO_ACIDS])
            else:
                features.extend([counts[aa] for aa in self.AMINO_ACIDS])
        return np.array(features)
    
    def extract_eaac(self, fasta_file, window=5, normalize=True, include_labels=False):
        """Extract EAAC features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        min_len = min(len(seq) for seq in sequences)
        n_windows = max(1, min_len - window + 1)
        features = []
        for seq in sequences:
            feats = self.eaac(seq, window, normalize)
            features.append(feats[:n_windows*len(self.AMINO_ACIDS)])
            
        feature_names = [
            f"EAAC_{aa}_w{i}" 
            for i in range(n_windows) 
            for aa in self.AMINO_ACIDS
        ]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== CKSAAP ====================
    def cksaap(self, sequence, k=0, normalize=True):
        """Composition of k-spaced Amino Acid Pairs"""
        pairs = [f"{a}{b}" for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        counts = {pair: 0 for pair in pairs}
        total = 0
        
        for i in range(len(sequence) - k - 1):
            pair = sequence[i] + sequence[i + k + 1]
            if pair in counts:
                counts[pair] += 1
                total += 1
        
        if normalize and total > 0:
            return np.array([counts[pair]/total for pair in pairs])
        return np.array([counts[pair] for pair in pairs])
    
    def extract_cksaap(self, fasta_file, k_max=5, normalize=True, include_labels=False):
        """Extract CKSAAP features for k=0 to k_max"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        pairs = [f"{a}{b}" for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        features = []
        
        for seq in sequences:
            seq_features = []
            for k in range(k_max + 1):
                seq_features.extend(self.cksaap(seq, k, normalize))
            features.append(seq_features)
        
        feature_names = [
            f"CKSAAP_{pair}_k{k}" 
            for k in range(k_max + 1) 
            for pair in pairs
        ]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== DPC ====================
    def dpc(self, sequence, normalize=True):
        """Dipeptide Composition"""
        dipeptides = [f"{a}{b}" for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        counts = {dp: 0 for dp in dipeptides}
        total = 0
        
        for i in range(len(sequence) - 1):
            dp = sequence[i:i+2]
            if dp in counts:
                counts[dp] += 1
                total += 1
        
        if normalize and total > 0:
            return np.array([counts[dp]/total for dp in dipeptides])
        return np.array([counts[dp] for dp in dipeptides])
    
    def extract_dpc(self, fasta_file, normalize=True, include_labels=False):
        """Extract DPC features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        dipeptides = [f"{a}{b}" for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        features = [self.dpc(seq, normalize) for seq in sequences]
        feature_names = [f"DPC_{dp}" for dp in dipeptides]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== DDE ====================
    def dde(self, sequence):
        """Dipeptide Deviation from Expected mean"""
        dipeptides = [f"{a}{b}" for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        aa_counts = {aa: 0 for aa in self.AMINO_ACIDS}
        dp_counts = {dp: 0 for dp in dipeptides}
        
        # Count amino acids and dipeptides
        for aa in sequence:
            if aa in aa_counts:
                aa_counts[aa] += 1
        
        for i in range(len(sequence) - 1):
            dp = sequence[i:i+2]
            if dp in dp_counts:
                dp_counts[dp] += 1
        
        # Calculate DDE features
        features = []
        total_aa = max(1, sum(aa_counts.values()))
        total_dp = max(1, sum(dp_counts.values()))
        
        for dp in dipeptides:
            observed = dp_counts[dp] / total_dp
            expected = (aa_counts[dp[0]] / total_aa) * (aa_counts[dp[1]] / total_aa)
            features.append((observed - expected) / expected if expected > 0 else 0)
        
        return np.array(features)
    
    def extract_dde(self, fasta_file, include_labels=False):
        """Extract DDE features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        dipeptides = [f"{a}{b}" for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        features = [self.dde(seq) for seq in sequences]
        feature_names = [f"DDE_{dp}" for dp in dipeptides]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== TPC ====================
    def tpc(self, sequence, normalize=True):
        """Tripeptide Composition"""
        tripeptides = [f"{a}{b}{c}" for a in self.AMINO_ACIDS 
                      for b in self.AMINO_ACIDS 
                      for c in self.AMINO_ACIDS]
        counts = {tp: 0 for tp in tripeptides}
        total = 0
        
        for i in range(len(sequence) - 2):
            tp = sequence[i:i+3]
            if tp in counts:
                counts[tp] += 1
                total += 1
        
        if normalize and total > 0:
            return np.array([counts[tp]/total for tp in tripeptides])
        return np.array([counts[tp] for tp in tripeptides])
    
    def extract_tpc(self, fasta_file, normalize=True, include_labels=False):
        """Extract TPC features"""
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
            
        tripeptides = [f"{a}{b}{c}" for a in self.AMINO_ACIDS 
                      for b in self.AMINO_ACIDS 
                      for c in self.AMINO_ACIDS]
        features = [self.tpc(seq, normalize) for seq in sequences]
        feature_names = [f"TPC_{tp}" for tp in tripeptides]
        
        if include_labels:
            return ids, np.array(features), feature_names, labels
        return ids, np.array(features), feature_names

    # ==================== Unified Interface ====================
    def extract_features(self, fasta_file, methods, params, include_labels=False):
        """
        Unified feature extraction interface
        :param fasta_file: Input FASTA file
        :param methods: List of methods to use (AAC, PAAC, EAAC, CKSAAP, DPC, DDE, TPC)
        :param params: Additional parameters for specific methods
        :param include_labels: Whether to include automatically assigned labels
        :return: (ids, features, feature_names) or (ids, features, feature_names, labels)
        """
        all_features = []
        all_names = []
        ids = None
        labels = None
        
        if include_labels:
            ids, sequences, labels = self.read_fasta(fasta_file, include_labels=True)
        else:
            ids, sequences = self.read_fasta(fasta_file)
        
        if 'AAC' in methods:
            result = self.extract_aac(fasta_file, include_labels=include_labels)
            if include_labels:
                _, features, names, _ = result
            else:
                _, features, names = result
            all_features.append(features)
            all_names.extend(names)
        
        if 'PAAC' in methods:
            lambda_val = params.get('lambda_val', 5)
            w = params.get('w', 0.05)
            result = self.extract_paac(fasta_file, lambda_val, w, include_labels=include_labels)
            if include_labels:
                _, features, names, _ = result
            else:
                _, features, names = result
            all_features.append(features)
            all_names.extend(names)
        
        if 'EAAC' in methods:
            window = params.get('window', 5)
            result = self.extract_eaac(fasta_file, window, include_labels=include_labels)
            if include_labels:
                _, features, names, _ = result
            else:
                _, features, names = result
            all_features.append(features)
            all_names.extend(names)
        
        if 'CKSAAP' in methods:
            k_max = params.get('k_max', 5)
            result = self.extract_cksaap(fasta_file, k_max, include_labels=include_labels)
            if include_labels:
                _, features, names, _ = result
            else:
                _, features, names = result
            all_features.append(features)
            all_names.extend(names)
        
        if 'DPC' in methods:
            result = self.extract_dpc(fasta_file, include_labels=include_labels)
            if include_labels:
                _, features, names, _ = result
            else:
                _, features, names = result
            all_features.append(features)
            all_names.extend(names)
        
        if 'DDE' in methods:
            result = self.extract_dde(fasta_file, include_labels=include_labels)
            if include_labels:
                _, features, names, _ = result
            else:
                _, features, names = result
            all_features.append(features)
            all_names.extend(names)
        
        if 'TPC' in methods:
            result = self.extract_tpc(fasta_file, include_labels=include_labels)
            if include_labels:
                _, features, names, _ = result
            else:
                _, features, names = result
            all_features.append(features)
            all_names.extend(names)
        
        # Combine all features
        combined_features = np.hstack(all_features) if all_features else np.array([])
        
        if include_labels:
            return ids, combined_features, all_names, labels
        return ids, combined_features, all_names
    
    def to_csv(self, fasta_file, methods, params, include_labels=False):
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