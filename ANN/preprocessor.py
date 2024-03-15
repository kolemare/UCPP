import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


class DataVisualizer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, delimiter=',', nrows=None)
        self.df.dataframeName = csv_file.split('/')[-1]

    def plot_per_column_distribution(self, n_graph_shown, n_graph_per_row):
        nunique = self.df.nunique()
        df = self.df[[col for col in self.df if nunique[col] > 1 and nunique[
            col] < 50]]  # For displaying purposes, pick columns that have between 1 and 50 unique values
        n_row, n_col = df.shape
        column_names = list(df)
        n_graph_row = (n_col + n_graph_per_row - 1) / n_graph_per_row
        plt.figure(num=None, figsize=(6 * n_graph_per_row, 8 * n_graph_row), dpi=80, facecolor='w', edgecolor='k')
        for i in range(min(n_col, n_graph_shown)):
            plt.subplot(int(n_graph_row), int(n_graph_per_row), int(i + 1))
            column_df = df.iloc[:, i]
            if not np.issubdtype(type(column_df.iloc[0]), np.number):
                value_counts = column_df.value_counts()
                value_counts.plot.bar()
            else:
                column_df.hist()
            plt.ylabel('counts')
            plt.xticks(rotation=90)
            plt.title(f'{column_names[i]} (column {i})')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        plt.show()

    def plot_correlation_matrix(self, graph_width):
        df = self.df.dropna(axis='columns')  # drop columns with NaN

        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df = df[numeric_cols]

        df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
        if df.shape[1] < 2:
            print(
                f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
            return
        corr = df.corr()
        plt.figure(num=None, figsize=(graph_width, graph_width), dpi=80, facecolor='w', edgecolor='k')
        corr_mat = plt.matshow(corr, fignum=1)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.gca().xaxis.tick_bottom()
        plt.colorbar(corr_mat)
        plt.title(f'Correlation Matrix for {self.df.dataframeName}', fontsize=15)
        plt.show()

    def plot_scatter_matrix(self, plot_size, text_size):
        df = self.df.select_dtypes(include=[np.number])  # keep only numerical columns
        # Remove rows and columns that would lead to df being singular
        df = df.dropna(axis='columns')
        df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
        column_names = list(df)
        if len(column_names) > 10:  # reduce the number of columns for matrix inversion of kernel density plots
            column_names = column_names[:10]
        df = df[column_names]
        ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plot_size, plot_size], diagonal='kde')
        corrs = df.corr().values
        for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
            ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',
                              va='center', size=text_size)
        plt.suptitle('Scatter and Density Plot')
        plt.show()

    def count_nan_values(self):
        nan_counts = self.df.isna().sum()
        print("Number of NaN values per column:")
        print(nan_counts)
        return nan_counts

    def load_data(self):
        n_row, n_col = self.df.shape
        print(f'There are {n_row} rows and {n_col} columns in raw dataset.')

        # Get the 24 most popular manufacturers based on the number of listings
        top_24_manufacturers = self.df['manufacturer'].value_counts().head(24).index.tolist()

        # Filter the DataFrame to include only the top 24 manufacturers
        self.df = self.df[self.df['manufacturer'].isin(top_24_manufacturers)]

        # Get the top 10 models for each of the top 24 manufacturers
        top_models = self.df.groupby('manufacturer')['model'].value_counts().groupby(level=0).head(10).reset_index(level=1, name='count')

        # Filter the original DataFrame to keep only the selected top models
        self.df = self.df.merge(top_models, on=['manufacturer', 'model'], how='inner')

        return self.df

    def load_and_describe_data(data):
        print("------------------------------------------------------------")
        print("Top 10 models per manufacturer")
        # Group the DataFrame by manufacturer and then get the top 10 models for each
        top_models_per_manufacturer = df.groupby('manufacturer')['model'].value_counts().groupby(level=0).head(10)

        # Print the top 10 models for each manufacturer along with their counts
        print(top_models_per_manufacturer)
        print("------------------------------------------------------------")
        print("Sum of models")
        # Sum the counts of the top 10 models for each manufacturer
        sum_top_models_per_manufacturer = top_models_per_manufacturer.groupby(level=0).sum()

        # Sort the sum of the top 10 models for each manufacturer in descending order
        sorted_sum_top_models_per_manufacturer = sum_top_models_per_manufacturer.sort_values(ascending=False)


        # Print the sum of the top 10 models for each manufacturer
        print(sum_top_models_per_manufacturer)

        print("------------------------------------------------------------")
        # Print the sorted sum of the top 10 models for each manufacturer
        print("Sorted sum of models (descending):")
        print(sorted_sum_top_models_per_manufacturer)

        total_sum_of_top_models = sum_top_models_per_manufacturer.sum()
        print("Total sum of top models:", total_sum_of_top_models)

        print("------------------------------------------------------------")

    def test_method(self):
        visualizer = DataVisualizer('../udataset/vehicles.csv')
        df1 = visualizer.load_and_describe_data()
        columns_to_remove = ['id', 'url', 'region_url', 'cylinders', 'title_status', 'VIN', 'size', 'paint_color', 'image_url', 'description', 'county']
        df1 = df1.drop(columns=columns_to_remove)
        df1 = df1.dropna()
        print(df1.head(100))
        print(f"Total number of rows in the DataFrame: {len(df1)}")
        visualizer.plot_per_column_distribution(10, 5)
        visualizer.plot_correlation_matrix(8)
        visualizer.plot_scatter_matrix(20, 10)
        visualizer.count_nan_values()


