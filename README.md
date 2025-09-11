# Infrang (INFormation Retrieval and ANswer Generation)

## 0. Contents

[1. Overview](#1-overview)  
[2. Features](#2-features)  
[3. Models](#3-models)  
[4. Architecture](#4-architecture)  
[5. Setup](#5-setup)  
[6. Use](#6-use)  
[7. Containerized version](#7-containerized-version)  
[8. Contributing](#8-contributing)  
[9. License](#9-license)  

## 1. Overview

Infrang is a Python class designed to provide methods on retrieving and generating information. It is particularly useful for RAG-based (Retrieval Augmented Generation) applications, or in other words, to combine searching for information with generating responses based on that information.  
With Infrang, you can build a knowledge base that not only retrieves relevant information but also generates insightful answers, making it a powerful tool for developers and researchers alike.
  
## 2. Features

*   **Multiple Interfaces**: Infrang offers both a Command-Line Interface (CLI) and a REST API, making it easy for users to interact with the system in a way that suits their preferences.
*   **Modular Design**: Infrang is built with a modular architecture, allowing easy extension and customization of its components.
*   **Hybrid Search**: The class supports Reciprocal Rank Fusion (RRF) combining dense and sparse embedding models, enabling the power of hybrid search.
*   **Text Preprocessing**: Infrang includes automated text preprocessing capabilities, including extraction and chunking.
*   **Multiple sources support**: A great variety of sources can be used for the retrieval part, such as PDF files, text files (`csv`, `json`, `md`, `txt` and `xml`), office files (`docx`, `xlsx` and `pptx`) and links (`url` and `urls`).
*   **Self-Contained Vector Database**: Infrang includes a fully managed, local embedded [QDrant](https://qdrant.tech/) vector database. This means no complex setup with Docker or external servers. Everything works out-of-the-box. All data is stored persistently in a local `./data` directory.
*   **Query rewriting**: The class has additional text processing capabilities such as word spelling control, paraphrasing or text expansion.
*   **Answer Generation**: Infrang generates answers based on the retrieved context, using [Groq API](https://console.groq.com/home).

## 3. Models

Infrang supports various models for different tasks:

- **Dense Model**: Default is `BAAI/bge-small-en-v1.5`.
- **Sparse Model**: Default is `prithivida/Splade_PP_en_v1`.
- **Paraphrase Model**: Default is `ramsrigouthamg/t5_paraphraser`.
- **Generative Model**: Default is `llama-3.3-70b-versatile`.

The dense and sparse models are the models that are used by QDrant for the retrieval part. The paraphrase model is lightweight (220M parameters) and thus it can be hosted locally and also provides fast inference responses. Finally the generative model is hosted on Groq Cloud so every call is done through an API.  
An API key for accessing the LLMs hosted in Groq is required and can be obtained [here](https://console.groq.com/keys).  
You can specify different models according to your specific use case. Note however, that the dense and sparse models should be supported by [QDrant](https://qdrant.tech/) and the generative model by [Groq](https://console.groq.com/home).  
Although the `Dense Model`, the `Sparse Model` and the `Generative Model` are required for the application to work, the `Paraphrase Model` is completely optional. The core RAG functionality will work even if this model is disabled (set to `None`)

## 4 Architecture

- **Document Processing**: You provide a folder of documents. Infrang automatically extracts text from them and splits it into smaller chunks.

- **Vectorization**: Each chunk is converted into numerical vectors (embeddings) using the configured dense and sparse models.

- **Storage**: These vectors are stored and indexed in a local, self-managed Qdrant database, ready for fast retrieval.

- **Querying**: When ythe user asks a question, it is also vectorized. Qdrant performs a hybrid search to find the most relevant text chunks.

- **Answer Generation**: These relevant chunks are sent to a powerful Groq-hosted LLM (like Llama 3), which generates a sourced answer from.

## 5. Setup

### 5.1 Create the base project

Open a terminal window, copy the repo to your path and navigate to it:
```bash
git clone https://github.com/christos-golsouzidis/infrang.git
cd infrang
```

<u>**File description**</u>

* `infrang_core.py`: This is the core library which contains the **Infrang** class.
* `infrang.py`: It contains the **CLI version**.
* `infrang-api.py`: It contains the **REST API** implemented with FastAPI.
* `requirements.txt`: All dependencies are contained here.

### 5.2 Create and activate a virtual environment

To avoid conflicts with system-wide Python packages, create and activate a virtual environment:

1. **Create a virtual environment**:
   ```bash
   python -m venv env
   ```
   or
   ```bash
   python3 -m venv env
   ```
   You can replace `env` with your desired environment name.

2. **Activate it**:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

### 5.3 Configure the Groq API key

1. Navigate to https://console.groq.com/keys and create a groq API key if you don't have one.

2. Infrang uses `python-dotenv`. In the `.env` file are all environment variables such as API keys. The Groq API key can be stored here by executing the following line:
```bash
echo "GROQ_API_KEY=<your-api-key-here>" > .env
```
or open a text editor, enter the line `GROQ_API_KEY=<your-api-key-here>` and save the file as `.env` at the current directory.  
Replace `<your-api-key-here>` with your actual groq API key.

**Note**: You can bypass this step and manually enter your Groq API key when prompted.

### 5.4 Install the required dependencies

To use Infrang, install via pip the required dependencies:

```bash
pip install -r requirements.txt
```
It might take some minutes as installing `torch` or other libraries can take a while.

## 6. Use

### 6.1 General
Infrang is a RAG application. Thus, the user must provide the path to the knowledge base as the first argument for both CLI version and the API version. The knowledge base is a folder containing all the files where the answer should be retrieved from. Supported filetypes are currently: `pdf`, `docx`, `xlsx`, `pptx`, `csv`, `url`, `urls`, `json`, `md`, `txt` and `xml`. Files with other extensions will be ignored. The path of the knowledge base can be anywhere regardless of the path where the database is created.  
**Note:**  
The `.url`, `.urls` files contain one or more links separated by '\n'. Each link is considered as a different source (although they are contained in a single file).  
Here is how a `.url` / `.urls` file should be like:

- `example.urls`
   ```example.urls
   https://en.wikipedia.org/wiki/Black_hole
   https://en.wikipedia.org/wiki/Stellar_black_hole
   https://en.wikipedia.org/wiki/Supermassive_black_hole
   ```
There is no distinction between the `.url` and the `.urls` file extension as Infrang process them in the same way. Thus the links can be stored in either filetype. For consistency it is recommended to use `.urls` only.
  
For more details see the paragraph [6.4 Complete example](#64-complete-example) below.

### 6.2 CLI Documentation

Click [here](#63-api-documentation) to go to the API documentation.  

#### 6.2.1 Quick start
To run the Infrang CLI, use the following command structure:

```bash
python infrang.py <collection> [options]
```

Replace `<collection>` with your collection name.
  
#### 6.2.2 CLI operations
- **Create a new collection / database**
   ```bash
   python infrang.py <collection> -c <path/to/knowledge_base>
   ```  

   *Create a new collection by replacing `<collection>` with your collection name and `<path/to/knowledge_base>` with the path to your directory where the document sources are located.*

- **Create and overwrite the existing collection / database**
   ```bash
   python infrang.py <collection> -c <path/to/knowledge_base> -o
   ```
   *The `-o` option creates a new and overwrites the collection if it already exists. This switch should be only used with `-c`*

- **Update the collection / database**
   ```bash
   python infrang.py <collection> -u <path/to/knowledge_base>
   ```

   *Updates the existing collection. Replace `<collection>` with your collection name and `<path/to/knowledge_base>` with the path to your directory where the new sources are located. The `<path/to/knowledge_base>` can (but doesn't have to) be the same to the path that the database was created. See the complete example [here](#64-complete-example).* 

- **Delete the collection / database**
   ```bash
   python infrang.py <collection> -d
   ```
   *Deletes the existing collection. Replace `<collection>` with your collection name.*

- **Perform a RAG operation according to the collection / database**
   ```bash
   python infrang.py <collection> -q <query>
   ```
   *Generates an answer according to the sources stored in the collection. This is equivalent to:*  
      ```bash
      python infrang.py <collection>
      ```
   *If `-q` is not provided, the user will be prompted to input a query. Replace `<collection>` and `<query>` with your collection name and your query respectively.*

- **Perform a RAG operation showing additional information about usage**
   ```bash
   python infrang.py <collection> -q <query> -v
   ```
   *The `-v` option shows additional information about the generated answer. This switch should be only used with `-q`*
  
See below the complete list of the valid arguments.

#### 6.2.3 Command-Line Arguments:


| Argument                    | Type   | Required | Default Value   | Description          |
|-----------------------------|--------|----------|-----------------|----------------------|
| `collection`                | string | Yes      | `default_collection`                   | The collection name.
| `-c`, `--create`            | string | No       | N/A                                    | The path to the knowledge base.
| `-u`, `--update`            | string | No       | N/A                                    | The path to the knowledge base.
| `-d`, `--delete`            | flag   | No       | False                                  | Deletes the collection if set. This deletes only the database but not the files or the directory of the knowledge base.
| `-q`, `--query`             | string | No       | N/A                                    | The query for searching retrieving and answering.
| `-lc`, `--list-collections` | flag   | No       | False                                  | Lists all collection names.
| `-ls`, `--list-sources`     | string | No       | N/A                                    | Lists all sources of a collection.
| `-dm`, `--dense_model`      | string | No       | `BAAI/bge-small-en-v1.5`               | The dense model to use for retrieval.
| `-sm`, `--sparse_model`     | string | No       | `prithivida/Splade_PP_en_v1`           | The sparse model to use for retrieval.
| `-pm`, `--paraphrase_model` | string | No       | `ramsrigouthamg/t5_paraphraser`        | The model to use for paraphrasing.
| `-gm`, `--generative_model` | string | No       | `llama-3.3-70b-versatile`              | The model to use for generative purposes.
| `-p`, `--parallel`          | int    | No       | 4                                      | Number of processes for storing to the database.
| `-o`, `--overwrite`         | flag   | No       | False                                  | Overwrites the existing database if set.
| `-v`, `--verbose`           | flag   | No       | False                                  | Shows additional information about the generated answer if set.
| `-g`, `--groq`              | string | No       | Uses GROQ_API_KEY from the environment (e.g., set via a .env file). If not provided, it prompts for input.   | The GROQ API key.
| `-de`, `--debug`            | flag   | No       | False                                  | Shows debugging information if set.


### 6.3 API Documentation

Click [here](#62-cli-documentation) to go to the CLI documentation.  

#### 6.3.1 Quick start
To run the Infrang API do the following steps:

Open a terminal window on the project folder having the virtual environment activated and execute the following line:
```bash
python infrang-api.py
```
The terminal should show some informations like:  
`Uvicorn running on http://127.0.0.1:7456 (Press CTRL+C to quit)`

Open another terminal window and execute a `curl` command (see below). Alternatively, you can use [Postman](https://www.postman.com/downloads/) to pass the API requests.

#### 6.3.2 Endpoints

* `GET /`  
   *Returns the name and the version of the application.*
   ```bash
   curl -X GET "http://127.0.0.1:7456/"
   ```

* `GET /collections`  
   *Returns the existing collections.*  
   ```bash
   curl -X GET "http://127.0.0.1:7456/collections"
   ```

* `GET /collections/{collection}`  
   **Parameters:**
   - `collection` (path): Name of the collection.  

   *Returns the sources contained in the specific collection.*  
   ```bash
   curl -X GET "http://127.0.0.1:7456/collections/my_collection"
   ```

* `POST /collections/{collection}/{path}`  
   **Parameters:**
   - `collection` (path): Name of the collection to create.
   - `path` (path): File system path to the documents directory.
   - `overwrite` (query, optional): Whether to overwrite existing collection (default: false).
   - `config` (body, optional): Configuration object (see the InfrangConfig class in `infrang-api.py`).  

   *Creates a new collection from the knowledge base given by the specified path.*  
   ```bash
   curl -X POST "http://127.0.0.1:7456/collections/my_collection//path/to/Knowledge_base" \
   -H "Content-Type: application/json" \
   -d '{
      "dense_model_name": "BAAI/bge-small-en-v1.5",
      "sparse_model_name": "prithivida/Splade_PP_en_v1",
      "parallel": 2
   }'
   ```
   *Note: The `-d` option is optional. You can completely ignore passing body parameters, as infrang will set the keys with their default values:*
   ```bash
   curl -X POST "http://127.0.0.1:7456/collections/my_collection//path/to/Knowledge_base"
   ```  
   *And here is how to overwrite the existing collection:*  
   ```bash
   curl -X POST "http://127.0.0.1:7456/collections/my_collection//path/to/Knowledge_base?overwrite=True"

* `PUT /collections/{collection}/{path}`  
   **Parameters:**
   - `collection` (path): Name of the collection to update
   - `path` (path): File system path to the documents directory
   - `config` (body, optional): Configuration object (see InfrangConfig model)  

   *Updates the collection from the knowledge base regarding the specified path.*  
   ```bash
   curl -X PUT "http://127.0.0.1:7456/collections/my_collection//path/to/Knowledge_base"
   ```

* `DELETE /collections/{collection}`  
   **Parameters:**
   - `collection` (path): Name of the collection to delete
   - `config` (body, optional): Configuration object (see InfrangConfig model)  

   *Deletes the specified collection.*  
   ```bash
   curl -X DELETE "http://127.0.0.1:7456/collections/my_collection/"
   ```

* `POST /answer/{collection}`  
   **Parameters:**
   - `collection` (path): The collection name of the database to generate an answer from.
   - `query` (query, required): The user's query.
   - `config` (body, optional): Configuration object (see the InfrangConfig model).  

   *Performs retrieval and generation operations.*  
   ```bash
   curl -X POST "http://127.0.0.1:7456/answer/my_collection?query=What+is+Python?" \
   -H "Content-Type: application/json" \
   -d '{
      "parallel": 4
   }'
   ```

### 6.4 Complete Example  

This end-to-end example demonstrates how to:  
1. Prepare a knowledge base (with mixed file types).  
2. Create/update a Qdrant collection.  
3. Query answers via CLI and API.  
4. Delete the collection at the end.

***Use case:*** *The user wants to learn Python from scratch.*

#### 6.4.1 Set Up the Knowledge Base

Create a folder named `python_docs` to your desired directory. Here we'll use `~` or `/home/user` as the base directory:
```bash
cd ~/Documents
mkdir python_docs
```
Let's add a PDF file as a source:  
- `python_learn.pdf`:  
   | | | 
   | --- | --- |
   | **Title:**  | Python for Everybody, Exploring Data Using Python 3  
   | **Description:**  | An awesome book for those who want to learn the fundamentals of Python.  
   | **Author:**  | Dr. Charles R. Severance  
   | **License:** | [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-nc-sa/3.0/)

Add it using the following line:

```bash
curl https://do1.dr-chuck.com/pythonlearn/EN_us/pythonlearn.pdf --output python_docs/python_learn.pdf
```
Finally navigate to the Infrang directory. Do the setup if you haven't already done it (see above the [setup](#5-setup)).

#### 6.4.2 CLI workflow

For the API workflow goto [6.4.3 API workflow](#643-api-workflow)

<u>**Create the database:**</u>  

Firstly we need to create a new collection and hence a database. Here the collection is named `python_101`.
```bash  
python infrang.py python_101 -c ~/Documents/python_docs
```
The `-c` option is for creating a database and takes as an argument a string of the path to the knowledge base.  
If everything is ok a collection will be added to the database.  

<u>**Generate an answer:**</u>  

Next type:  
```bash  
python infrang.py python_101
```
Use the above syntax for querying multiple questions. For asking a simple question you can use this syntax:
```bash
python infrang.py python_101 -q "Ask your question here"
```
which is the same as the command above. So if you don't specify the query with the `-q` parameter, you will be prompted to enter your query.

*Output:*
> Ask me something or press 'Enter' to exit:

If an empty query is given, Infrang CLI terminates.  

Type the following question:  
*Query:*
```plaintext  
What is a constructor?
```  
You will get a response like this:  
*Output:*  
> A constructor is a specially named method (usually `__init__`) that is called when an object is constructed, used to set up initial values for the object.

Let's try another question:  
*Query:*  
```plaintext  
What is a decorator?
```  
You will get a response like this:  
*Output:*
> I don't know the answer.

This is expected because the source doesn't contain any information about *decorators*. Thus, we have to update our collection / database with another source which contains that information.  
Press *Enter* to terminate Infrang CLI (input is an empty string) and return to the terminal.  

<u>**Update the database:**</u>  

This time we will add a link as a source, which will be stored in a `.urls` file.
  
Type:
```bash  
echo "https://www.geeksforgeeks.org/python/decorators-in-python/" > ~/Documents/python_docs/decorators.urls
```  
This line creates a `.urls` file containing the link we provide it. You can check its contents with:  
```bash
cat ~/Documents/python_docs/decorators.urls
```
If you see the URL then everything went well.  
Next write:
```bash
python infrang.py python_101 -u ~/Documents/python_docs
```
The `-u` option is for updating the database.  
Reask the question; this time with the `-q` option:  
```bash
python infrang.py python_101 -q "What is a decorator?"
```  
This time you will get a response like this:  
**Output:**  
> Decorators are a flexible way to modify or extend the behavior of functions or methods in Python, without changing their actual code. They are essentially functions that take another function as an argument and return a new function with enhanced functionality.  

To acquire additional informations about the generated answer add the `-v` switch:
```bash
python infrang.py python_101 -q "What is a decorator?" -v
```  
**Output:**  
> Decorators are a flexible way to modify or extend the behavior of functions or methods in Python, without changing their actual code. They are essentially functions that take another function as an argument and return a new function with enhanced functionality.  
{
   "completion_time": 0.028722356,
   "prompt_time": 0.100944339,
   "total_time": 0.129666695,
   "completion_tokens": 22,
   "prompt_tokens": 1191,
   "total_tokens": 1213
}

<u>**Remove the Collection:**</u>  

To remove the `python_101` collection run this command:
```bash
python infrang.py python_101 -d
```

#### 6.4.3 API Workflow  

For the CLI workflow goto [6.4.2 CLI workflow](#642-cli-workflow)

<u>**Start the API:**</u>  

To start the API enter this line:
```bash  
python infrang-api.py
```  
then open a new terminal. From now on type here all of the following commands, but feel free to use *Postman* instead if you like.  

<u>**Checking the API:**</u>  

If you visit `http://127.0.0.1:7456/` which is the same as making a GET request:
```bash  
curl -X GET "http://127.0.0.1:7456"
```
and you see something like 
> {"message":"Infrang API, version 1.2.4"}  

the API works.

<u>**Create the database:**</u>  

Firstly we need to create a new collection and hence a database. Here the collection is named `python101`.
```bash  
curl -X POST "http://127.0.0.1:7456/collections/python101//home/user/Documents/python_docs"
```
Note that the path we passed is `/home/user/Documents/python_docs`, which is a path parameter. Passing `%2Fhome%2Fuser%2FDocuments%2Fpython_docs` would be also acceptable, since FastAPI decodes `%2F` to `/`.  
If everything is ok a collection will be added to the database and the response would be:
> {"message":"Database created successfully",
"collection":"python101",
"path":"/home/user/Documents/python_docs"}

<u>**Generate an answer:**</u>  

Type the following command:  
```bash
curl -X POST "http://127.0.0.1:7456/answer/python101?query=What+is+Python?"
```
Here we passed the question `What is Python?` as a query parameter and we encoded *whitespace* characters with `+`.  
You will get a response like this:  
> {"collection":"python101","query":"What is Python?","result":{"answer":"Python is a high-level language intended to be relatively straightforward for humans to read and write and for computers to read and process.","usage":{"completion_time":0.064736186,"prompt_time":0.111010902,"total_time":0.175747088,"completion_tokens":26,"prompt_tokens":1409,"total_tokens":1435}}}

Let's try the question `What is a decorator?`:  
```bash
curl -X POST "http://127.0.0.1:7456/answer/python101?query=What+is+a+decorator?"
```

You will get a response like this:  
> {"collection":"python101","query":"What is a decorator?","result":{"answer":"I don't know the answer.","usage":{"completion_time":0.010001355,"prompt_time":0.111424087,"total_time":0.121425442,"completion_tokens":8,"prompt_tokens":1420,"total_tokens":1428}}}

So the value of the `response['result']['answer']` is `"I don't know the answer"`. This is expected because the source doesn't contain any information about *decorators*. Thus, we have to update our collection / database with another source which contains that information.  

<u>**Update the database:**</u>  

This time we will add a link as a source, which will be stored in a `.urls` file.

```bash  
echo "https://www.geeksforgeeks.org/python/decorators-in-python/" > ~/Documents/python_docs/decorators.urls
```  
This line creates a `.urls` file containing the link we provide it. You can check its contents with:  
```bash
cat ~/Documents/python_docs/decorators.urls
```
If you see the URL then everything went well.  
Now we have to update our collection which is done by entering the following:
```bash
curl -X PUT "http://127.0.0.1:7456/collections/python101//home/user/Documents/python_docs"
```

Let's resend the request querying `What is a decorator?`:  
```bash
curl -X POST "http://127.0.0.1:7456/answer/python101?query=What+is+a+decorator?"
```

This time you will get a response like this:  
> {"collection":"python101","query":"What is a decorator?","result":{"answer":"A decorator is a function that takes another function as an argument and returns a new function with enhanced functionality.","usage":{"completion_time":0.028722356,"prompt_time":0.100944339,"total_time":0.129666695,"completion_tokens":22,"prompt_tokens":1191,"total_tokens":1213}}}

<u>**Remove the Collection:**</u>  

To remove the `python101` collection send this request:
```bash
curl -X DELETE "http://127.0.0.1:7456/collections/python101"
```
You will get this response:
> {
    "message": "Collection deleted successfully",
    "collection": "python101"
}

## 7 Containerized version

The API version of Infrang can be executed locally as a container. This project utilizes [Podman](https://podman.io/), an open-source tool licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). However, if you prefer to use [Docker](https://www.docker.com/), only minor adjustments are needed.

<u>**Benefits of Containerization:**</u>

- **Automated Dependency Management**: All dependencies are defined in a Dockerfile, allowing for easy installation with a single command.
- **Portability**: The application can run on any system with Podman / Docker installed, ensuring compatibility across different operating systems.
- **Isolation**: Each container runs in its own environment, preventing conflicts between applications.

### 7.1 Install Podman / Docker

To run the containerized version of Infrang API you need to install [Podman](https://podman.io/docs/installation) if you use Podman for containerizing the app, or [Docker](https://docs.docker.com/engine/install/) if you intend to use Docker. 

### 7.2 Build the image

First of all, build the image based on the `Dockerfile`.  
If you use Podman write this line:  
```bash
podman build -t infrang-api .
```
or in Docker:  
```bash
docker build -t infrang-api .
```

**Note** The Dockerfile remains the same either with Podman or with Docker.  

### 7.3 Create the volume

Next, create the volume for persistent data storage:  
```bash
podman volume create infrang-data
```
or Docker:  
```bash
docker volume create infrang-data
```

### 7.4 Run the container

Ensure you have created and configured your `.env` file for the application's environment variables.

The run commands differ slightly between the two tools. The `:Z` flag in the Podman command is required on Linux systems with SELinux enabled to properly label the volume mount.

Enter this if using Podman:
```bash
podman run -d --name infrang-api \
-p 7456:7456 \
--env-file .env \
-v infrang-data:/data/:Z \
-v <my-base-path>:/var/lib/base/:Z \
infrang-api
```
or this if using Docker:
```bash
docker run -d --name infrang-api \
-p 7456:7456 \
--env-file .env \
-v infrang-data:/data \
-v <my-base-path>:/base \
infrang-api
```
Replace `<my-base-path>` with your **absolute** base path.  
E.g. use this mounting line for Linux: `-v /home/user/my_data:/base:Z`  
or for Windows: `-v C:\Users\user\data:/base`.

**Important note on paths:** The path inside the container (`/base`) becomes the root for all API calls. For example, if you mount your local folder `/home/john/Documents` to `/base`, then to process the documents located at `/home/john/Documents/my_kb`, you would use the path **`/base/my_kb`** in your API requests (e.g.
`curl -X POST "http://127.0.0.1:7456/collections/python101//base/my_kb"`).


### 7.5 Automate the process

For Linux users, the bash script `infrang-podman.sh` automates the process even further. By running the script, you can have a running container. So this automates the steps 7.2, 7.3 and 7.4. You still have to stpo the container with `podman stop infrang-api` at the end.

## 8 Contributing

Contributions to Infrang are welcome! Please submit pull requests or issues on the GitHub repository.

## 9 License

Infrang is licensed under the [MIT License](https://opensource.org/licenses/MIT).
