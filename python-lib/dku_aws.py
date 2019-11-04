import collections
import boto3
import pandas as pd

def aws_client(service_name, connection_info):
    return boto3.client(service_name=service_name, aws_access_key_id=connection_info.get('accessKey'), aws_secret_access_key=connection_info.get('secretKey'), region_name=connection_info.get('region'))

def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return new_name
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")

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
