from pathlib import Path
from llama_index import download_loader
from llama_index import SimpleDirectoryReader
import os

#from llamaidx_gai import document_loader_func
from document_loaders import document_loader_func

def get_meta(file_path):
  _, extension = os.path.splitext(file_path)
  if extension in [".txt", ".json", ".htm", ".html"]:
       return {"page_label": "1","file_name": os.path.basename(file_path)}
  return {"file_name": os.path.basename(file_path)}

@document_loader_func(name="custom_sdr", description="customized SDR")
def custom_sdr(inputs, reader):

   loader_args = inputs.get("loader_args", {})
   data_args = inputs.get("data_args", {})

   dir_reader = SimpleDirectoryReader(**loader_args,file_metadata=get_meta)
   documents = dir_reader.load_data(**data_args)
   return documents
