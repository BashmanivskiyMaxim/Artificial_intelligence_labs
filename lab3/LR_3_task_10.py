import datetime
import json
import numpy as np
from sklearn import covariance, cluster
import yfinance as yf

# Input file containing company symbols
input_file = "lab3/company_symbol_mapping.json"

# Load the company symbol map
with open(input_file, "r") as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Define the date range for historical stock quotes
start_date = "2003-07-03"
end_date = "2007-05-04"

# Download historical stock quotes using yfinance
quotes = []
valid_symbols = []
for symbol in symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty:
            quotes.append(data)
            valid_symbols.append(symbol)
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")

# Check if there are valid symbols
if not quotes:
    print(
        "No valid data available for any symbol. Check your symbol mapping and data availability."
    )
else:
    symbols = valid_symbols  # Update symbols to valid ones

    # Extract opening and closing quotes
    opening_quotes = np.array([quote["Open"].values for quote in quotes]).T
    closing_quotes = np.array([quote["Close"].values for quote in quotes]).T

    # Compute differences between opening and closing quotes
    quotes_diff = closing_quotes - opening_quotes

    # Normalize the data
    X = quotes_diff.copy()
    X /= X.std(axis=0)

    # Create a graph model
    edge_model = covariance.GraphicalLassoCV()

    # Train the model
    with np.errstate(invalid="ignore"):
        edge_model.fit(X)

    # Build clustering model using Affinity Propagation model
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    num_labels = labels.max()

    # Print the results of clustering
    print("\nClustering of stocks based on difference in opening and closing quotes:\n")
    for i in range(num_labels + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_names = names[cluster_indices]
        if len(cluster_names) > 0:
            print("Cluster", i + 1, "==>", ", ".join(cluster_names))
