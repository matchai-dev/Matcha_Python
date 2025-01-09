import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
from flask import Flask, jsonify

app = Flask(__name__)

STATIC_FILE_PATH = './Data_500_Aged.csv'

def load_dataset(file_path):
    """Loads the dataset from the given file path."""
    return pd.read_csv(file_path)

def preprocess_categorical_columns(dataset):
    """Encodes categorical columns using MultiLabelBinarizer."""
    encoded_fields = []
    for field in dataset.columns[2:-1]:  # Assuming first two columns are 'User_Id' and 'Sex'
        field_data = dataset[field].fillna("").astype(str).str.split(", ")
        mlb = MultiLabelBinarizer()
        binary_data = mlb.fit_transform(field_data)
        binary_df = pd.DataFrame(binary_data, columns=[f"{field}_{option}" for option in mlb.classes_])
        encoded_fields.append(binary_df)
    return pd.concat([dataset[['User_Id', 'Sex', 'Age']]] + encoded_fields, axis=1)

def apply_one_hot_encoding(X):
    """Applies one-hot encoding to the first column."""
    ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    return np.array(ct1.fit_transform(X))

def scale_features(X):
    """Scales features in the dataset."""
    sc = StandardScaler()
    X[:, 2:3] = sc.fit_transform(X[:, 2:3])
    return X

def apply_pca(X, variance_ratio=0.95):
    """Applies PCA to reduce dimensionality."""
    pca = PCA(n_components=variance_ratio)
    return pca.fit_transform(X)

def apply_umap(X, n_components=2, n_neighbors=30, min_dist=0.0, spread=0.5, metric='euclidean'):
    """Applies UMAP for dimensionality reduction."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        random_state=42
    )
    return reducer.fit_transform(X)

def final_scaling(X):
    """Scales the final reduced features using MinMaxScaler."""
    final_scaler = MinMaxScaler()
    return final_scaler.fit_transform(X)

def cluster_data(X, eps=0.036, min_samples=5):
    """Clusters data using DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X)

def create_service():
    """Processes the dataset and returns user IDs, clusters, and sexes."""
    # Load dataset
    dataset = load_dataset(STATIC_FILE_PATH)

    # Preprocess categorical columns
    preprocessed_dataset = preprocess_categorical_columns(dataset)

    # Extract features
    X = preprocessed_dataset.iloc[:, 1:].values
    
    # Apply one-hot encoding and scaling
    X = apply_one_hot_encoding(X)
    X = scale_features(X)

    # Apply PCA
    X_pca = apply_pca(X)

    # Apply UMAP
    X_reduced = apply_umap(X_pca)

    # Final scaling
    X_final = final_scaling(X_reduced)

    # Cluster data
    y_dbscan = cluster_data(X_final)

    # Extract required information
    user_ids = preprocessed_dataset['User_Id'].values
    sexes = preprocessed_dataset['Sex'].values

    # Combine results into a structured DataFrame
    results = pd.DataFrame({"user_id": user_ids, "cluster": y_dbscan, "sex": sexes})
    return results

@app.route('/getmatches', methods=['GET'])
def get_matches():
    """API endpoint to retrieve user IDs, clusters, and sexes after retraining."""
    try:
        # Generate the service results by retraining and predicting
        results = create_service()

        # Format the results for JSON response
        response = []
        for _, row in results.iterrows():
            response.append({
                "user_id": row["user_id"],
                "cluster": int(row["cluster"]),
                "sex": row["sex"]
            })

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
