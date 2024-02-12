# Data Loaders
  The data loader configuration allows users to specify different sources for ingesting text documents into the system. Depending on the user's preference and the location of their data, they can choose from various sources and for which we have also provided the different yamls for the readers.
  You need to replace the reader configuration as per your use case in the ingestion.yaml file.

# File Loader
  This loader takes in a local directory containing files and extracts Documents from each of the files. By default, the loader will utilize the specialized loaders in this library to parse common file extensions (e.g. .pdf, .png, .docx, etc). For more info , you can visit "https://llamahub.ai/l/file?from=loaders"

# Simple Website Loader
  This loader is a simple web scraper that fetches the text from static websites by converting the HTML to text.
  For more info , you can visit "https://llamahub.ai/l/web-simple_web?from=loaders"

# Wikipedia Loader
 This loader fetches the text from Wikipedia articles . The inputs may be page titles or keywords that uniquely identify a Wikipedia page. In its current form, this loader only extracts text and ignores images, tables, etc.
 For more info , you can visit ""https://llamahub.ai/l/wikipedia?from=loaders"

# Microsoft SharePoint Loader
  The loader loads the files from a folder in sharepoint site. It also supports traversing recursively through the sub-folders.
  For more info , you can visit "https://llamahub.ai/l/microsoft_sharepoint?from=loaders"

