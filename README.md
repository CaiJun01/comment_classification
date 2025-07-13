# 🧠 Project Title: Comment Classification: True, False, or Opinion?

> This is a simple project that presents my vision of how AI can be useful to counter misinformation and falsehoods on social media platforms.

---

## 📌 Motivation

> As an active social media user, I mostly learn about current affairs through community-contributed comments on forums and social media posts. However, I am also constantly worried about the veracity of the information. I am often worried about deliberate falsehoods, unintentional spread of fake news, or incorrectly interpreting someone's opinion as a fact. Therefore, I imagined social media platforms to be equipped with an extension for users to conveniently perform fact-checking on what they consume. At the time of writting and as much as I know, only X has equipped their comment section with Grok, enabling their users to find out more about the contents of the comment. This project is the application of what I have learned about AI, NLP and LLMs from school. Despite it being far from perfect, I would like to share with the github community.


## 📌 Instructions for demo

> git clone https://github.com/caijun01/comment_classification.git
> cd Backend
> 
> python -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
>
> uvicorn main:app --reload
>
> Navigate to Frontend and open index.html with Live Server
>
> Hover over the second comment by animalfarmlover2 and click on the blue sparkling icon.
>
> Do be patient and wait for awhile as the text processing takes awhile
>
> Once the comment has been processed, the True statements will be highlighted in Green, Red for False statements, and Blue for opinions.

---

## 📌 Documentation of my project
> For fact-checking applications, our models need to be "trained" on a "ground-truth". In reality, it would be the internet and most models such as chatgpt and claude have already done so. However, instead of taking the easy way out and calling APIs, I decided to make my own "ground-truth". I decided on the book Animal Farm as it was a book that I am familiar with. I proceeded to download the Ebook (can be found under claim_classifier folder "orwellanimalfarm.pdf"). The claim_classifier.ipynb is a notebook where I wrote the codes to train the model on the facts of the book. I used pymupdf to read the pdf into a single string variable 'text'. I then cleaned up 'text' by removing the title of the book that appeared on every page and also the hyperlink "http://www.mudmap.com ... AM]". Next, I performed fixed-size chunking with chunk size = 200 and overlap = 50. I would have preferred to use semantic chunking over here because Animal Farm is a book novel after all and paragraph and chapter breaks become crucial considerations when chunking to retain to right meaning. However, because of how I performed the pdf reading, I was not able to retain the natural structure of the book that is important in order to use langchain' semantic chunking library. I did attempt to do so, but the results of the chunks were poor, with chunks being cut up in the middle of sentences. Hence, I defaulted to fixed size chunking with overlap. I saved the chunks and moved on the embed it. I used the sentence_transformers library and used the "BAAI/bge-base-en" embedder because the embedder was trained on retrieval-specific tasks and uses state-of-the-art similarity. I then saved the embeddings to be used later in main.py under the Backend folder.
>
> Once I have completed the first part, I can only implement my model to tell between True and False statements. However, some comments are opinionated and cannot be classified under either of them. Hence, I need to create another model that decides whether each statement is an opinion or claim. Opinion statements do not need to be further evaluated, whereas claims will be passed into the first model for True/False evaluation. Now, I need to develop an Opinion/Claim classifier. Under the fact_opinion_classifier, there is a fact_opinion_classifier.ipynb notebook and a ClaimBuster_Datasets folder. I am using labelled data generously provided by ClaimBuster (https://zenodo.org/records/3836810) to train a classifier that can tell whether a sentence is a claim or an opinion. My codes for the training of this classifier is in fact_opinion_classifier.ipynb notebook. I used DeBERTa as my classification model as it is known to outperform BERT and RoBERTa in natural language understanding tasks. The model training would have been impossible if not for google colab's GPU. An evaluation of the model on the test split of the dataset achieved a 93.1% accuracy and a 89% Precision score (I chose to evaulate on Precision as this context calls for minimising false positives, because it is more "costly" to misclassify an opinion as a claim since it might then be incorrectly evaluated as True when it is an opinion in fact.
>
> Once I have the two models prepared, I created main.py that can be found under the Backend folder. main.py is built with FastAPI and it received a comment, splits it up into sentences, and run each sentence through the fact_opinion_classifier model. If the statement is classified as a claim, it will proceed on to the second model to be evaluated against the animal farm book on whether the claim is True or False. For the evaluation, I wrote a code that encodes each statement and calculates a similarity score with all the chunks in the book. I then take the statement and the most similar chunk and pass it into a Natural Language Inference model. I had initially wanted to build my own NLI model but I had recognised the complexity of the task and deemed it too ambitious for this project. Hence, I used a NLI pipeline that used deberta tokenizer and model to evaluate whether my statement agrees with the most similar chunk or not. The NLI pipeline outputs a label (either True, False or Unknown). Initially, I noticed that the pipeline was extraordinarily good at identifying False claims, but it was very conservative when it comes to labelling a claim as 'True'. I tested some true claims and frequently returned 'Unknown'. I suppose the NLI is primarily used for fake news detection instead of entailment. Hence, modified the threshold for the NLI to determine if a statement is True, allowing it accurately classify my test cases.

## 📌 Gaps and Improvements
> The project is far from perfect. Ideally, I wanted to be able to perform semantic chunking on the novel to experience the impact that chunking has on such a use case. I had also hoped to implement a function that can identify phrases to be evaluated, instead of parsing each sentence into my model. This is because comments in reality might have multiple claims in a sentence that could be classified differently. In my attempt of extracting phrases from the comments, the results I got had repeated phrases or some of them turned out to be stemmed/lemmatized. For example, ['Napoleon is a horrible person', 'Napoleon horrible', 'horrible person']. Basically, I was not able to uniquely identify phrases for evaluation. Lastly, I would have loved to deploy this project on the web, but because this is a 'hobby' project, I cannot afford to pay for premium deployment tiers as required for my model tensors and embeddings.

## 📌 Tech Stack
> Backend: FastAPI, Python
> Models: DeBERTa, sentence-transformers
> Frontend: HTML, JavaScript, CSS
> Libraries: PyMuPDF, langchain, transformers
