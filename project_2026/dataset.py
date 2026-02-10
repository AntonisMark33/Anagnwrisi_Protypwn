import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CrimeDataset:
    def __init__(self, file_path):
        # Load the raw CSV data
        self.data = pd.read_csv(file_path)
        self.train = None
        self.val = None
        self.test = None
        # S is the number of distinct serial killers (hidden in data)
        self.S = self.data['killer_id'].nunique()

    def preprocess(self):
        """
        Decomposes features into continuous and categorical parts as per Section 2.1.
        d_c = 8 continuous features
        d_cat = 17 encoded categorical features (one-hot)
        """
        # Continuous descriptors (d_c=8)
        self.cont_cols = [
            'hour_float', 'latitude', 'longitude', 'victim_age',
            'temp_c', 'humidity', 'dist_precinct_km', 'pop_density'
        ]

        # Raw categorical variables to be encoded
        self.cat_raw_cols = ['weapon_code', 'scene_type', 'weather', 'vic_gender']

        # Perform One-Hot Encoding
        # This converts integer codes into binary vectors e^(weapon), e^(scene), etc.
        self.data_encoded = pd.get_dummies(self.data, columns=self.cat_raw_cols)

        # Identify the new one-hot columns
        self.cat_encoded_cols = [
            c for c in self.data_encoded.columns
            if any(c.startswith(raw + '_') for raw in self.cat_raw_cols)
        ]

    def split_train_val_test(self):
        """
        Splits data based on the 'split' column provided in the dataset.
        """
        self.train = self.data_encoded[self.data_encoded['split'] == 'TRAIN']
        self.val = self.data_encoded[self.data_encoded['split'] == 'VAL']
        self.test = self.data_encoded[self.data_encoded['split'] == 'TEST']

        # Keep raw versions for simple plotting or debugging if needed
        self.train_raw = self.data[self.data['split'] == 'TRAIN']
        self.val_raw = self.data[self.data['split'] == 'VAL']
        self.test_raw = self.data[self.data['split'] == 'TEST']