
import argparse
import getpass
from infrang_core import Infrang
import dotenv
import os


def generate_answer(infrang, query, debug, verbose):
    result = infrang.answer(query=query, debug=debug)
    print(result['answer']
    )
    if verbose:
        print(result['usage'])
    return


def main():
    
    parser = argparse.ArgumentParser(
                    prog='Infrang',
                    description='A RAG application (CLI version)')

    parser.add_argument('collection', type=str, help='The path to the knowledge base.')
    parser.add_argument('-c', '--create', type=str, required=False, help='Creates a new database.')
    parser.add_argument('-u', '--update', type=str, required=False, help='Updates the existing database.')
    parser.add_argument('-d', '--delete', action='store_true', help='Deletes the collection.')
    parser.add_argument('-ls', '--list-sources', action='store_true', help='Returns the sources of the collection.')
    parser.add_argument('-lc', '--list-collections', action='store_true', help='Returns the collections.')
    parser.add_argument('-q', '--query', type=str, required=False, help='Answers the query based on the collection.')
    parser.add_argument('-dm', '--dense_model', type=str, required=False, help='The dense model.',
                        default='BAAI/bge-small-en-v1.5')
    parser.add_argument('-sm', '--sparse_model', type=str, required=False, help='The sparse model.',
                        default='prithivida/Splade_PP_en_v1')
    parser.add_argument('-pm', '--paraphrase_model', type=str, required=False, help='The paraphrase model.',
                        default='ramsrigouthamg/t5_paraphraser')
    parser.add_argument('-gm', '--generative_model', type=str, required=False, help='The generative model.',
                        default='llama-3.3-70b-versatile')
    parser.add_argument('-p', '--parallel', type=int, required=False, default=4, 
                        help='Number of processes for storing to the database. Default value: 4.')
    parser.add_argument('-o', '--overwrite', action='store_true', 
                        help='Overwrites the existing database. Default value: False.')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Shows additional information about the generated answer. Default value: False.')
    parser.add_argument('-g', '--groq', required=False, 
                        help='The groq API key. If not provided, it uses GROQ_API_KEY stored in the virtual environment.')
    parser.add_argument('-de', '--debug', action='store_true', 
                        help='Shows debugging information. Default value: False.')

    args = parser.parse_args()
    dotenv.load_dotenv()

    if not args.groq and not dotenv.get_key('.env', 'GROQ_API_KEY'):
        groq_api_key = getpass.getpass('Enter Groq API key:')
    else:
        groq_api_key = None

    infrang = Infrang(collection=args.collection,
                dense_model_name=args.dense_model,
                sparse_model_name=args.sparse_model,
                paraphrase_model_name=args.paraphrase_model,
                generate_model_name=args.generative_model,
                parallel=args.parallel,
                groq_api_key=groq_api_key, # if None, it will be derived from the virtual environment
            )
    
    
    if args.create: # -c option
        infrang.create(kb_path=args.create, overwrite=args.overwrite)
        return
    elif args.update: # -u option
        infrang.update(kb_path=args.update)
        return
    elif args.delete: # -d option
        infrang.delete()
        return
    elif args.list_collections:
        _, collections, _ =  next(os.walk(os.path.join('data', 'collection')))
        print(collections)
        return
    elif args.list_sources:
        print(
            infrang.get_sources()
        )
        return
    elif args.query: # -q option
        if args.collection not in infrang.get_collections():
            print('Error: Collection does not exist.')
            return
        generate_answer(infrang, args.query, args.debug, args.verbose)
    else:  # no option set
        if args.collection not in infrang.get_collections():
            print('Error: Collection does not exist.')
            return

        while True:
            query = input('\nAsk me something or press \'Enter\' to exit:\n')
            if not query:
                return
            
            generate_answer(infrang, query, args.debug, args.verbose)

if __name__ == '__main__':
    main()

