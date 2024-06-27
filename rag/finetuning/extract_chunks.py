import mlflow
import requests
import pandas as pd
import csv
import os
import argparse
from mlflow.tracking import MlflowClient
import json
import subprocess
client = MlflowClient()
# Setting mlflow tracking uri
os.environ['MLFLOW_TRACKING_URI'] = "http://d3x-controller.d3x.svc.cluster.local:5000"

def retreive_chunks(vector_id_list, output_csv_path, output_chunks_json, experiment_name, no_of_chunks, cleaned_chunks_dir):
    # Specify your Weaviate server URL
    weaviate_url = "http://weaviate.d3x.svc.cluster.local"

    all_data = []
    all_data_json = []

    chunks_ft_path = "./temp_out/"
    output_json_path = f"./temp_out/0-000000/{output_chunks_json}"

    # Check if the directory exists
    directory = os.path.dirname(output_json_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if no_of_chunks is None:
        vector_ids_to_process = vector_id_list

    else:
        vector_ids_to_process = vector_id_list[:no_of_chunks]

    for vector_id in vector_ids_to_process:
        # Construct the URL for the object retrieval
        url = f"{weaviate_url}/v1/objects/{vector_id}"

        # Make a GET request to retrieve the object
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse and print the retrieved object
            retrieved_object = response.json()
            #paper_chunks = retrieved_object['properties']['paperchunks']
            paper_chunks = retrieved_object.get('properties', {}).get('paperchunks', '')
            all_data.append({'vector_id': vector_id, 'paper_chunks': paper_chunks})
            all_data_json.append({'chunks': paper_chunks})
        else:
            # Print an error message if the request was not successful
            print(f"Failed to retrieve object. Status code: {response.status_code}, Response: {response.text}")
    # Write the paper_chunks data to a CSV file
    #with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    #    csv_writer = csv.writer(csvfile)
    #    csv_writer.writerow(['vector_id','paper_chunks'])  # Write header
    #    csv_writer.writerows([[data['vector_id'], data['paper_chunks']] for data in all_data])
        # Log the CSV file as an artifact in MLflow
    # Write the paper_chunks data to a JSON file
    chunk_size = 500
    chunks = [all_data_json[i:i+chunk_size] for i in range(0, len(all_data_json), chunk_size)]
    #print(chunks)

    for i, chunk in enumerate(chunks):
        i_str = str(i).zfill(6)
        output_json_path = f"./temp_out/0-{i_str}/"
        if not os.path.exists(output_json_path):
            os.makedirs(output_json_path)
        with open(f"{output_json_path}/text_chunks.json", 'w', encoding='utf-8') as jsonfile:
            json.dump(chunk, jsonfile)
        print(f"Chunk {i} written to {output_json_path}")

    """
    chunks_dir_inc = 0
    for data in all_data_json:
        inc_str = str(chunks_dir_inc).zfill(6)
        output_json_path = f"./temp_out/0-{chunks_dir_inc}/./text_chunks.json"
        with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump([{'chunks': data['chunks']} for data in all_data_json], jsonfile)
            #    json.dump([{'chunks': data['chunks']} for data in all_data_json], jsonfile)
        chunks_dir_inc += 1
        print(chunks_dir_inc)

    """
    try:
        # Specify your shell command
        command = "your_shell_command_here"

        # Execute the shell command
        result = subprocess.run(f"d3x fm trainchunks --source {chunks_ft_path} --destination {cleaned_chunks_dir} ", shell=True, check=True, stdout=subprocess.PIPE)

        # If the command executed successfully, print the output
        print(result.stdout.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        print(f"Error creating train chunks: {e}")
    #mlflow.set_experiment(experiment_name)
    #with mlflow.start_run():
    #    mlflow.log_artifact(output_csv_path, artifact_path="weaviate_data")
    #    mlflow.log_artifact(output_chunks_json, artifact_path="weaviate_data_json")

def extract_column_values(csv_file_path, column_name):
    """
    Extract values from a specified column in a CSV file and return them as a list.

    Parameters:
    - csv_file_path (str): Path to the CSV file.
    - column_name (str): Name of the column to extract.

    Returns:
    - list: List of values from the specified column.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)

        # Extract the column data into a list
        column_values = df[column_name].tolist()

        return column_values

    except Exception as e:
        print(f"Error: {e}")
        return None

def artifacts_download(run_id,local_dir):

    artifact=client.download_artifacts(run_id,"",local_dir)

    print(f"Artifacts downloaded to: {local_dir}/chunks/")
    return local_dir

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process Weaviate data and log to MLflow.')
    parser.add_argument('--experiment_name', type=str,required=True ,help='MLflow experiment name')
    parser.add_argument('--run_id', type=str, required=True,help='MLflow run ID for artifact download')
    parser.add_argument("-d", "--destination", type=str, required=True, help="The path where chunks will be kept for training")
    parser.add_argument('--no_of_chunks', type=int,help="retreives the chunk text for first given number")
    args = parser.parse_args()

    csv_file_path = artifacts_download(args.run_id, ".")
    csv_file_path += "/chunks/chunks.csv"
    print(csv_file_path)
    column_name = "chunk_id"
    vector_ids_list = extract_column_values(csv_file_path, column_name)
    output_csv_file = "./retrieved_chunks.csv"
    output_chunks_json = "./text_chunks.json"
    print(f"output_csv_file saved to the following {output_csv_file}")
    print(f"output_chunks_json saved to the following {output_chunks_json}")
    retreive_chunks(vector_ids_list, output_csv_file, output_chunks_json, args.experiment_name, args.no_of_chunks, cleaned_chunks_dir=args.destination)
    print(f"output csv file also artifacted to mlflow with experiment name {args.experiment_name}")
if __name__ == "__main__":
    main()