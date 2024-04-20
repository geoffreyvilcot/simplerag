from config import Config
import os
import json
import numpy as np
import pickle
import getopt
import sys
import requests
import json
from tqdm import tqdm
from vector_db_manager import Vector_DB, Vector_DB_Qdrant
from llm_wrapper import Llm_wrapper


def process_file_text(conf : Config, model : Llm_wrapper, filename : str, max_tokens=256) -> [str] :
    chunks = []

    base_filename = os.path.basename(filename)
    with open(filename, "r", encoding='utf-8') as f:

        line = f.readline()
        current_chunk = f""
        while line :
            if len(current_chunk) > 1 :
                word_count = len(current_chunk.split(' '))
                if word_count>max_tokens:
                    if len(current_chunk) > 100:
                        chunks.append({'Description': current_chunk, 'filename':base_filename })
                        current_chunk = f""
            current_chunk += line
            line = f.readline()
        if len(current_chunk) > 10 :
            chunks.append({'Description': current_chunk, 'filename':base_filename })
    return chunks


def process_file(conf : Config, model : Llm_wrapper, filename : str)  :
    chunks = process_file_text(conf, model, filename, max_tokens=conf.ingest_max_tokens)

    return  chunks

if __name__ == '__main__':
    conf_file_name = "config.json"

    opts, args = getopt.getopt(sys.argv[1:],"hc:")
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -c <conf_file>')
            sys.exit()
        elif opt in ("-c"):
            conf_file_name = arg

    conf = Config(conf_file=conf_file_name)

    llm = Llm_wrapper(conf)


    input_files = [os.path.join(conf.ingest_files_dir, f) for f in os.listdir(conf.ingest_files_dir)]
    input_files = sorted(input_files)
    print(f"Total number of files {len(input_files)}")

    if conf.ingest_limit_files is not None and conf.ingest_limit_files > 0 :
        input_files = input_files[conf.ingest_start_file_index:conf.ingest_start_file_index+conf.ingest_limit_files]

    stack = []
    stack_chunks = []
    chunk_dic = {}
    print("Read file")
    for i in tqdm(range(len(input_files))) :
        filename = input_files[i]
        chunks = process_file(conf, llm, filename)
        stack_chunks.extend(chunks)

    # print(stack_chunks)
    print(len(stack_chunks))


    idx = 0
    array_emb = []
    # for chunk in stack_chunks :
    print("Compute Embeddings")
    db = None
    for i in tqdm(range(len(stack_chunks))) :
        chunk = stack_chunks[i]
        array_emb.append(llm.embed(chunk['Description']))

        idx +=1
    text_embeddings = np.array(array_emb)

    if db == None:
        d = text_embeddings.shape[1]
        print(f"dim : {d}")
        db = Vector_DB_Qdrant(conf, d)
        db.reset()

    db.add(text_embeddings, stack_chunks)
    db.save()

