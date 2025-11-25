from pathlib import Path
from llama_index import download_loader
import uuid
import os

from document_loaders import document_loader_func


def get_meta(file_path):
  _, extension = os.path.splitext(file_path)
  if extension in [".txt", ".json"]:
       return {"page_label": "1","file_name": os.path.basename(file_path)}
   return {"file_name": os.path.basename(file_path)}

@document_loader_func(name="S3Loader", description="loader to load files from s3")
def S3Loader(inputs, reader):
   S3Reader = download_loader('S3Reader', custom_path="/tmp")

   loader_args = inputs.get("loader_args", {})
   data_args = inputs.get("data_args", {})

   UnstructuredReader = download_loader('UnstructuredReader', custom_path="/tmp")

   loader = S3Reader(
   bucket =        loader_args['bucket'],
   aws_access_id = loader_args['aws_access_id'],
   aws_access_secret = loader_args['aws_access_secret'],
   file_metadata = get_meta,
   file_extractor={
     ".pdf": UnstructuredReader(),
   }    
   )

   documents = loader.load_data(**data_args)
   for d in documents: 
       d.doc_id = str(uuid.uuid4())
   return documents
