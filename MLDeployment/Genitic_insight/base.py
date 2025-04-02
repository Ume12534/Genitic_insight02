import os
import csv
from io import StringIO
import numpy as np
from Bio import SeqIO
from collections import Counter

class FeatureExtractor:
    @staticmethod
    def read_fasta(file_path):
        """Read fasta file and return sequences and IDs"""
        sequences = []
        ids = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq).upper())
            ids.append(record.id)
        return ids, sequences

    @staticmethod
    def to_csv(ids, features, feature_names):
        """Convert features to CSV string"""
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        # Write header
        writer.writerow(['ID'] + feature_names)
        
        # Write data rows
        for seq_id, feature_vec in zip(ids, features):
            writer.writerow([seq_id] + list(feature_vec))
        
        csv_data = csv_buffer.getvalue()
        csv_buffer.close()
        return csv_data