
import os
from markitdown import MarkItDown
import markdownify as mdf
from pdfminer.high_level import extract_text as mine_text
from qdrant_client import models, QdrantClient
import torch
from spellchecker import SpellChecker
from transformers import T5Tokenizer, T5ForConditionalGeneration
from groq import Groq
from urllib.parse import urlparse
import time
import requests


class Infrang:
    '''
        INFormation Retrieval and ANswer Generation: A class to be used by RAG applications.

        Methods:
            __init__
            get_sources
            create
            update
            delete
            answer
    '''

    def __init__(self,
                collection,
                dense_model_name='BAAI/bge-small-en-v1.5',
                sparse_model_name='prithivida/Splade_PP_en_v1',
                paraphrase_model_name='ramsrigouthamg/t5_paraphraser',
                generate_model_name='llama-3.3-70b-versatile',
                parallel=4,
                groq_api_key=None,
                ):
        '''
        Initializes the Infrang instance with the specified document path and model configurations.
            Params: 
                **collection (str):** Name of the collection.
                **sparse_model_name (str):** Name of the sparse embedding model.
                **dense_model_name (str):** Name of the dense embedding model.
                **paraphrase_model_name (str):** Name of the paraphrasing model.
                **generate_model_name (str):** Name of the generating model that is used by Groq.
                **parallel (int):** Number of parallel processes for database operations. Default value is 4.
                **groq_api_key (str):** API key for Groq service. If not provided, it uses GROQ_API_KEY stored in the virtual environment.
        '''
        
        self.DESTINATION_SOURCES = '__sources.list'
        self.collection = collection or 'default_collection'
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.parallel = parallel
        self.database_client = None
        self.__setup_init()
        self.paraphrase_tokenizer = None
        self.paraphrase_model = None
        if paraphrase_model_name:
            self.paraphrase_tokenizer = T5Tokenizer.from_pretrained(paraphrase_model_name, legacy=False)
            self.paraphrase_model = T5ForConditionalGeneration.from_pretrained(paraphrase_model_name)
        self.generate_model = generate_model_name
        if groq_api_key:
            self.groq = Groq(
                    api_key=groq_api_key
                )
        else:
            self.groq = Groq()


    def __setup_init(self):
        if not os.path.exists('data'):
            os.makedirs('data')
            time.sleep(0.05)
        try:
            self.database_client = QdrantClient(path='data')
        except:
            print('Closing existing database instanse...')
            self.database_client.close()
            time.sleep(0.1)
            self.database_client = QdrantClient(path='data')


    def get_collections(self):

        _, dirs, _ =  next(os.walk(os.path.join('data', 'collection')))
        return dirs
    

    def uninit(self):
        '''
            Uninitializes the Infrang instance and closes the connection to QDrant DB
        '''
        self.database_client.close()


    def __etl(self, kb_dir, doc):

        def __extract_text(src: str):

            def extract_pypdf(src):
                return mine_text(src)

            def extract_markitdown(src):
                extractor = MarkItDown()
                return extractor.convert(src).text_content

            def extract_plain(src):
                with open(src) as fr:
                    return fr.read()

            result = urlparse(src)
            if result.netloc and result.scheme:
                time.sleep(0.1) # avoid getting 403
                response = requests.get(src, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }).content
                return mdf.markdownify(response), src
            if src.endswith('.pdf'):
                return extract_pypdf(os.path.join(kb_dir, src)), src
            elif src.endswith(('.docx','.xlsx','.pptx','.csv')):
                return extract_markitdown(os.path.join(kb_dir, src)), src
            elif src.endswith(('.json', '.md', '.txt', '.xml', '.ini')):
                return extract_plain(os.path.join(kb_dir, src)), src
            else:
                print('Skipping source {} : Filetype not supported. Only these filetypes are supported:' \
                'pdf, docx, xlsx, pptx, csv, url, urls, json, md, txt, xml.\n'\
                .format(src))
                return None, None
        
        def __chunk_text(text, source, length=1200, overlap=200):

            chunks = []
            offset = 0
            while True:
                chunks.append(
                    {
                        'text' : text[offset:offset+length],
                        'source' : source
                    })
                if offset + length >= len(text):
                    break # if it captures the text until the end no need to iterate further
                offset += length - overlap
            return chunks
        
        def __upsert(metadata):

            existing = self.database_client.count(self.collection, exact=True).count
            self.database_client.upload_collection(
                collection_name=self.collection,
                vectors=[
                    {
                        'dense': models.Document(text=doc['text'], model=self.dense_model_name),
                        'sparse': models.Document(text=doc['text'], model=self.sparse_model_name)
                    } for doc in metadata],
                payload=metadata,
                parallel=self.parallel,
            )
            total = self.database_client.count(self.collection, exact=True).count
            print(
            'Added {} new entries; Total entries: {}'.format(total - existing, total)
            )

        text, source = __extract_text(doc)
        if not text: 
            return False
        print(f'Processing {source} :')
        print('Chunking text...')
        metadata = __chunk_text(text=text, source=source)
        print('Storing data...')
        __upsert(metadata=metadata)
        return True


    def __get_current_sources(self, kb_path, files):

        src = []
        for file in files:
            if file.endswith('.url') or file.endswith('.urls'):
                with open(os.path.join(kb_path, file)) as fr:
                    for link in set(fr.read().splitlines()):
                        if not link:
                            continue
                        src.append(link)
            else:
                src.append(file)
        return set(src)


    def __get_existing_sources(self):

        with open(os.path.join('data', 'collection', self.collection, self.DESTINATION_SOURCES), 'r') as f:
            return set(f.read().splitlines())


    def __update_existing_sources(self, src):
        
        with open(os.path.join('data', 'collection', self.collection, self.DESTINATION_SOURCES), 'a') as f:
            f.write(src + '\n')


    def get_sources(self):
        '''
            Returns a list with the sources of the collection.
        '''

        return list(self.__get_existing_sources())
    

    def create(self, kb_path, overwrite=False):
        '''
            Creates a new Qdrant collection for storing document embeddings if it does not already exist.
                Params:
                    **kb_path**: Path to the knowledge base
                    **overwrite (bool):** If true, it replaces the sources that already exist with the new. Default value is False.
        '''

        kb_dir, _, files = next(os.walk(kb_path)) # the directory of the knowledge base
        base_path = os.path.join('data', 'collection', self.collection, self.DESTINATION_SOURCES)
        base_dir = os.path.join('data', 'collection', self.collection)
        current_docs = self.__get_current_sources(kb_dir, files)

        if not overwrite:
            if os.path.exists(base_path):
                print('The database exists already.')
                return
        else:
            self.database_client.delete_collection(collection_name=self.collection)
            time.sleep(0.1)

        print('Creating database...')
        os.makedirs(base_dir)
        with open(base_path, 'x'):
            pass
        self.database_client.create_collection(
            collection_name=self.collection,
            vectors_config={'dense': models.VectorParams(
                    size=self.database_client.get_embedding_size(self.dense_model_name), 
                    distance=models.Distance.COSINE
                )},
            sparse_vectors_config={'sparse': models.SparseVectorParams()},
        )
        
        for doc in current_docs:
            self.__etl(kb_dir, doc)
            self.__update_existing_sources(doc)

        print('Done!')


    def update(self, kb_path):
        '''
            Updates the database given the knowledge base.
                Params:
                    **kb_path**: Path to the knowledge base
        '''

        kb_dir, _, files = next(os.walk(kb_path)) # the directory of the knowledge base

        try:
            new_docs = self.__get_current_sources(kb_dir, files) - self.__get_existing_sources()
        except FileNotFoundError as e:
            print(e)
            print('Creating a new collection...')
            self.create(kb_path=kb_dir, overwrite=False)
            return

        if not new_docs:
            print(
                'Warning: There are no new documents to update.'
            )
            return

        for doc in new_docs:
            if not self.__etl(kb_dir, doc):
                continue
            self.__update_existing_sources(doc)

        print('Done!')


    def delete(self):
        '''
            Deletes the collection
        '''
        _, collections, _ = next(os.walk(os.path.join('data','collection')))
        if self.collection in collections:
            self.database_client.delete_collection(collection_name=self.collection)
            print('Collection removed successfully.')
        else:
            print('Error: Could not find the collection to remove it.')


    def answer(self, query: str, debug=False):
        '''
        Performs a semantic search over the stored documents using dense and sparse models, and generates an answer based on the retrieved context.
            Params:
                **query (str):** The query string to search for in the database.
            Returns:
                A dictionary containing the generated answer and usage statistics.
        '''

        def search(query: str, limit=8):
            assert type(query) == str
            search_result = self.database_client.query_points(
                collection_name=self.collection,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF
                ),
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query, model=self.dense_model_name),
                        using='dense',
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model=self.sparse_model_name),
                        using='sparse',
                    ),
                ],
                query_filter=None,
                limit=limit,
            )
            return [{
                'metadata': result.payload,
                'score': result.score,
            }
            for result in search_result.points]

        def get_response(query, num_responses=1, max_length=64):
            batch = self.paraphrase_tokenizer.encode_plus(query, padding=True, return_tensors="pt")
            with torch.no_grad():
                translated = self.paraphrase_model.generate(**batch,
                    max_length=max_length,
                    num_beams = num_responses,
                    num_return_sequences=num_responses)
                return self.paraphrase_tokenizer.batch_decode(translated, skip_special_tokens=True)

        def check_spelling(query, distance=1):
            spell = SpellChecker(distance=distance)
            words = query.strip().split()
            # If a word contains at least one upper case character or is inside quotes ignore correction for this word
            corrected_words = [spell.correction(word) or word 
                            if (word.islower() or word[0] == "'" or word[0] == '"') else word 
                            for word in words]
            return " ".join(corrected_words)
            
        def generate(query: str, context: list[str], model=self.generate_model):
            assert type(context) == list
            system_prompt = '''
You are an assistant that answers questions strictly based on the CONTEXTS below.
Do not use external knowledge or guess. If the answer is missing, say: "I don't know the answer."
Keep responses concise (1-2 sentences unless more detail is needed).
'''
            system_prompt += ''.join(['\n\n<CONTEXT>\n' + item + '\n</CONTEXT>' for item in context])
            
            response = self.groq.chat.completions.create(
                messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                ],
                model=model,
            )
            return {
                'answer' : response.choices[0].message.content,
                'usage' : {
                    'completion_time': response.usage.completion_time,
                    'prompt_time': response.usage.prompt_time,
                    'total_time': response.usage.total_time,

                    'completion_tokens': response.usage.completion_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens,
                }
            }
        
        if not query:
            return
        query = check_spelling(query)
        if self.paraphrase_model and self.paraphrase_tokenizer:
            query = get_response(query)[0]
            if debug:
                print('<rewrite>\n{}\n</rewrite>\n'.format(query))
        results = search(query, limit=4)
        if debug:
            for num, result in enumerate(results):
                print('<{} result>\n{}\n</result>\n'.format(num, result))
        text_results = [item['metadata']['text'] for item in results]
        return generate(query=query, context=text_results)

