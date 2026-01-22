import pandas as pd
from sklearn.model_selection import train_test_split

class CrimeDataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.train = None
        self.val = None
        self.test = None
        self.S = self.data['killer_id'].nunique()

    def preprocess(self):
        # The assignment mentions dc=8 continuous features
        self.cont_cols = ['hour_float', 'latitude', 'longitude', 'victim_age', 'temp_c', 'humidity', 'dist_precinct_km', 'pop_density']
        self.cat_raw_cols = ['weapon_code', 'scene_type', 'weather', 'vic_gender']
        
        # One-hot encode categorical variables
        # We need to keep the original values for some tasks or use a specific encoding
        self.data_encoded = pd.get_dummies(self.data, columns=self.cat_raw_cols)
        # Store the list of encoded columns
        self.cat_encoded_cols = [c for c in self.data_encoded.columns if any(c.startswith(raw + '_') for raw in self.cat_raw_cols)]

    def split_train_val_test(self):
        self.train = self.data_encoded[self.data_encoded['split']=='TRAIN']
        self.val   = self.data_encoded[self.data_encoded['split']=='VAL']
        self.test  = self.data_encoded[self.data_encoded['split']=='TEST']
        
        # Also keep non-encoded version for some tasks
        self.train_raw = self.data[self.data['split']=='TRAIN']
        self.val_raw   = self.data[self.data['split']=='VAL']
        self.test_raw  = self.data[self.data['split']=='TEST']

    def plot_histograms(self, *, bins=30, figsize=(10, 6), show=True, block=False, save_path=None):
        """
        Plot histograms for continuous columns.

        - show=True: display a window
        - block=False: do not block script execution (recommended for scripts)
        - save_path: if provided, saves the figure and closes it (great for batch runs)
        """
        import matplotlib.pyplot as plt

        cont_cols = ['hour_float', 'victim_age', 'latitude', 'longitude']
        ax_arr = self.data[cont_cols].hist(bins=bins, figsize=figsize)
        fig = ax_arr[0, 0].get_figure() if hasattr(ax_arr, "shape") else plt.gcf()

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return

        if show:
            plt.show(block=block)
            if not block:
                # Give the GUI loop a moment to draw; keeps things responsive in scripts/IDEs
                plt.pause(0.001)

    def plot_2d_scatter(self, x_col, y_col, *, alpha=0.5, show=True, block=False, save_path=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.data[x_col], self.data[y_col], alpha=alpha)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return

        if show:
            plt.show(block=block)
            if not block:
                plt.pause(0.001)
