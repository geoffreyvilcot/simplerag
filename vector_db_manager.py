import pickle
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

import os

class Vector_DB :
    def __init__(self, conf : Config, dim):
        pass

    def reset(self):
        pass

    def add(self, vector, payload):
        pass

    def search(self, vector, k=3):
        pass

    def save(self):
        pass
    def load(self):
        pass

class Vector_DB_Qdrant(Vector_DB) :
    def __init__(self, conf : Config, dim=None):
        super().__init__(conf, dim)
        self.conf = conf
        if self.conf.qdrant_local :
            self.client = QdrantClient(path=self.conf.vector_db_file)
        else :
            self.client = QdrantClient(self.conf.qdrant_host, port=self.conf.qdrant_port)
        self.k_vector = 3
        self.dim = dim

    def reset(self):
        if not self.client.collection_exists(collection_name=self.conf.qdrant_collection) :
            self.client.recreate_collection(
                collection_name=self.conf.qdrant_collection,
                vectors_config=VectorParams(size= self.dim, distance=Distance.DOT),
        )

    def add(self, vector, payload):
        vector_points = []
        for i in range(vector.shape[0]) :
            # vector_points.append(PointStruct(id=i, vector=vector[i].tolist(), payload={"data": payload[i]}))
            vector_points.append(PointStruct(id=i, vector=vector[i].tolist(), payload=payload[i]))
            if len(vector_points) > 100 :
                operation_info = self.client.upsert(
                    collection_name=self.conf.qdrant_collection,
                    wait=True,
                    points=vector_points,
                )
                vector_points = []
                print(operation_info)

        if len(vector_points) > 0 :
            operation_info = self.client.upsert(
                collection_name=self.conf.qdrant_collection,
                wait=True,
                points=vector_points,
            )
            vector_points = []
            print(operation_info)

    def search(self, vector, k=3):
        search_result = self.client.search(
            collection_name=self.conf.qdrant_collection, query_vector=vector[0].tolist(), limit=k
        )
        # print(search_result)
        retrieve_payloads = [ result.payload for result in search_result]
        return retrieve_payloads
        # D, I = self.index.search(vector, k=self.k_vector)  # distance, index
        #
        # retrieved_payloads = [self.payloads[i] for i in I.tolist()[0]]
        # return retrieved_payloads


