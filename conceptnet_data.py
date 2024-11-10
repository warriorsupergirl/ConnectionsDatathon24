import dask.dataframe as dd

# Path to the ConceptNet CSV file
conceptnet_file_path = "assertions.csv"

# Load the CSV file using Dask, including all columns
conceptnet_df = dd.read_csv(
    conceptnet_file_path,
    sep='\t',
    header=None,
    names=['uri', 'relation', 'start', 'end', 'weight', 'dataset', 'sources'],
    blocksize="64MB"  # Adjust block size for tuning (e.g., 64MB, 128MB)
)
# Filter to only include rows where the 'start' and 'end' columns contain English concepts
conceptnet_df = conceptnet_df[(conceptnet_df['start'].str.contains('/c/en/')) & 
                              (conceptnet_df['end'].str.contains('/c/en/'))]
# Display the first few rows to check the data structure
print(conceptnet_df.head())
row_10 = conceptnet_df.compute().iloc[10]
start_value = row_10['start']
end_value = row_10['end']
print(f"Start: {start_value}, End: {end_value}")



#This work includes data from ConceptNet 5, which was compiled by the Commonsense Computing Initiative. ConceptNet 5 is freely available under the Creative Commons Attribution-ShareAlike license (CC BY SA 4.0) from https://conceptnet.io. The included data was created by contributors to Commonsense Computing projects, contributors to Wikimedia projects, Games with a Purpose, Princeton University's WordNet, DBPedia, OpenCyc, and Umbel.
