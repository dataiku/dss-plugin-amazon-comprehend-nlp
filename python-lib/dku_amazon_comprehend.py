import boto3

ALL_ENTITY_TYPES = ['COMMERCIAL_ITEM', 'DATE', 'EVENT', 'LOCATION', 'ORGANIZATION', 'OTHER', 'PERSON', 'QUANTITY', 'TITLE']

def get_client(connection_info):
    return boto3.client(service_name='comprehend', aws_access_key_id=connection_info.get('accessKey'), aws_secret_access_key=connection_info.get('secretKey'), region_name=connection_info.get('region'))

def format_sentiment_results(raw_results):
    sentiment = raw_results.get('Sentiment').lower()
    output_row = dict()
    output_row["raw_results"] = raw_results
    output_row["predicted_sentiment"] = sentiment
    score = raw_results.get('SentimentScore',{}).get(sentiment.capitalize())
    output_row["predicted_probability"] = round(score, 2)
    return output_row

def format_language_results(raw_results):
    output_row = dict()
    output_row["raw_results"] = raw_results
    if len(raw_results.get('Languages')):
        language = raw_results.get('Languages')[0]
        output_row["detected_language"] = language.get('LanguageCode')
        output_row["probability"] = round(language.get('Score'), 2)
    else:
        output_row["detected_language"] = ''
        output_row["probability"] = ''
    return output_row

def format_keyphrases_results(raw_results):
    output_row = dict()
    output_row["raw_results"] = raw_results
    output_row["keyphrases"] = _distinct([kp["Text"] for kp in raw_results.get("KeyPhrases", [])])
    return output_row

def format_entities_results(raw_results):
    output_row = dict()
    output_row["raw_results"] = raw_results
    output_row["entities"] = [_format_entity(e) for e in raw_results.get("Entities", [])]
    for t in ALL_ENTITY_TYPES:
        output_row[t] = _distinct([e["text"] for e in output_row["entities"] if e["type"] == t])
    return output_row

def _format_entity(e):
    return {
        "type": e.get("Type"),
        "text": e.get("Text"),
        "score": e.get("Score"),
        "beginOffset": e.get("BeginOffset"),
        "endOffset": e.get("EndOffset"),
    }

def _distinct(l):
    return list(dict.fromkeys(l))
