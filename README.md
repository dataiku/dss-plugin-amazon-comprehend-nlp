# Amazon Comprehen


## Plugin information

This Dataiku DSS plugin provides several tools to interact with [Amazon Rekognition](https://aws.amazon.com/comprehend/), the NLP (Natural Language Processing) API.

The API can be used to analyze unstructured text for tasks such as sentiment analysis, key phrase extraction and language detection.
No training data is needed to use this API; just bring your text data. This API uses advanced natural language processing techniques to deliver best in class predictions.
[Read the documentation](https://aws.amazon.com/comprehend/) for more information.


## Using the Plugin

### Prerequisites
In order to use the Plugin, you will need:

* an AWS account
* proper [credentials](https://docs.aws.amazon.com/comprehend/latest/dg/setting-up.html) (access tokens) to interact with the service:
* make sure you know in **which AWS region the services are valid**, the Plugin will need this information to get authenticated

### Plugin components
The Plugin has the following components:

    * [Sentiment Analysis](https://docs.aws.amazon.com/comprehend/latest/dg/how-sentiment.html):
    evaluates text input and returns a sentiment score for each document, ranging from 0 (negative) to 1 (positive). This capability
    is useful for detecting positive and negative sentiment in social media, customer reviews, and discussion forums.
    Content is provided by you; models and training data are provided by the service.
    * [Key Phrases Extraction](https://docs.aws.amazon.com/comprehend/latest/dg/how-key-phrases.html):
    evaluates unstructured text, and for each JSON document, returns a list of key phrases. This capability is useful if you need to quickly
    identify the main points in a collection of documents. For example, given input text "The food was delicious and there were wonderful staff",
    the service returns the main talking points: "food" and "wonderful staff".
    * [Named Entity Recognition](https://docs.aws.amazon.com/comprehend/latest/dg/how-entities.html):
    takes unstructured text, and for each JSON document, returns a list of disambiguated entities with links to more information on the web (Wikipedia and Bing).

