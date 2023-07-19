import weaviate

#WEAVIATE_URL = "http://a2ad89a5a36444dd8ace283b1f8c1a5f-1765265645.us-east-2.elb.amazonaws.com"
WEAVIATE_URL = "http://localhost:30716/api/vectordb/"
#WEAVIATE_URL = "http://localhost:32197/api/vectordb/"

client = weaviate.Client(url=WEAVIATE_URL, 
                        additional_headers={  # (Optional) Any additional headers; e.g. keys for API inference services
                            "Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZGVmYXVsdCIsInR5cGUiOiJ1c2VyIiwiaWQiOiI1OWRlZWI5Yy1lOTkxLTRiZjItYTZlNS04YTc1YTNiZjZiNDkifQ.10fune-o5hIzioprNrF-L57dOOl0M-j3O95kzZRYdHc"
                         })

schema = client.schema.get()
print(schema)

all_classes = [c["class"] for c in schema["classes"]]
print(f"Classes: {all_classes}")

r = client.query.aggregate(class_name='Dankitairdocs').with_meta_count().do()
print(r)
#n_docs = r["data"]["Aggregate"]["Document"][0]["meta"]["count"]
#print(f"Number of documents: {n_docs}")


r = client.query.aggregate(class_name='Dankitairchunks').with_meta_count().do()
print(r)
#n_chunks = r["data"]["Aggregate"]["Chunk"][0]["meta"]["count"]
#print(f"Number of chunks: {n_chunks}")

print("------------")
print()
batch_size = 10
class_name = "Dankitairchunks"
class_properties = ["name","doc", "paperchunks"]
cursor = None


def get_batch_with_cursor(client, class_name, class_properties, batch_size, cursor=None):

    query = (
        client.query.get(class_name, class_properties)
        # Optionally retrieve the vector embedding by adding `vector` to the _additional fields
        #.with_additional(["id vector"])
        .with_additional(["id "])
        .with_limit(batch_size)
    )

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()

results = get_batch_with_cursor(client, class_name, class_properties, batch_size, cursor)
print(results)


print("------------")
print()

#client.schema.delete_class("Article") # deletes the class "Article" along with all data points of class "Article"
# OR
#client.schema.delete_all() # deletes all classes along with the whole data

batch_size = 10
class_name = "Dankitairdocs"
class_properties = ["paperdoc", "dockey"]
cursor = None


def get_batch_with_cursor(client, class_name, class_properties, batch_size, cursor=None):

    query = (
        client.query.get(class_name, class_properties)
        # Optionally retrieve the vector embedding by adding `vector` to the _additional fields
        #.with_additional(["id vector"])
        .with_additional(["id "])
        .with_limit(batch_size)
    )

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()

results = get_batch_with_cursor(client, class_name, class_properties, batch_size, cursor)
print(results)
