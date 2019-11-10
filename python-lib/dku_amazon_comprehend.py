import collections
import boto3
import pandas as pd

def get_client(connection_info):
    return boto3.client(service_name='comprehend', aws_access_key_id=connection_info.get('accessKey'), aws_secret_access_key=connection_info.get('secretKey'), region_name=connection_info.get('region'))

def group_by_language(batch, text_column, language_column):
    text_by_language = collections.defaultdict(list)
    original_index_by_language = collections.defaultdict(list)
    for index, row in batch.iterrows():
        text = row.get(text_column, '')
        if pd.isna(text):
            text = ''
        language = row.get(language_column)
        if text != '' and language:
            text_by_language[language].append(text)
            original_index_by_language[language].append(index)
        elif text == '' and language:
            text_by_language[language].append(' ')
            original_index_by_language[language].append(index)
        elif text != '' and not language:
            text_by_language['en'].append(text)
            original_index_by_language['en'].append(index)
        else:
            text_by_language['en'].append(' ')
            original_index_by_language['en'].append(index)
    return text_by_language, original_index_by_language

def batch_detect_sentiment(batch, client, text_column, language_column):
    text_by_language, original_indices_by_language = group_by_language(batch, text_column, language_column)
    results_per_language = collections.defaultdict(list)
    for language, request in text_by_language.items():
        re = client.batch_detect_sentiment(TextList=request, LanguageCode=language)
        results_per_language[language] = re.get('ResultList')
    result_per_row = [None] * len(batch)
    for language, original_indices in original_indices_by_language.items():
        for i, original_index in enumerate(original_indices):
            result_per_row[original_index] = results_per_language[language][i]

    output_rows = []
    for original_index, row in batch.iterrows():
        text = row[text_column]
        result = result_per_row[original_index]
        if text:
            sentiment = result.get('Sentiment').lower()
            row[predicted_sentiment_column] = sentiment
            if output_probabilities:
                row[predicted_probability_column] = result.get('SentimentScore',{}).get(sentiment.capitalize())
            if should_output_raw_results:
                row["raw_results"] = json.dumps(result)
        output_rows.append(row)
    return output_rows
