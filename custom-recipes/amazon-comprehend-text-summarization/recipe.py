import logging
import collections
import dataiku
from dataiku.customrecipe import *
from common import *
from dku_amazon_comprehend import *

BATCH_SIZE = 10

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[comprehend plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get('connectionInfo', {})
text_column = get_recipe_config().get('text_column', None)
language_column = get_recipe_config().get('language_column', 'en')
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role('input-dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()

output_dataset_name = get_output_names_for_role('output-dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

output_schema = input_schema
output_column_name = generate_unique(text_column + "_summary", [col['name'] for col in input_schema])
output_schema.append({'name': output_column_name, 'type':'string'},)
output_dataset.write_schema(output_schema)
if should_output_raw_results:
    output_schema.append({"name": "raw_results", "type": "string"})
writer = output_dataset.get_writer()

for batch in input_dataset.iter_dataframes(chunksize=BATCH_SIZE):
    request_data = collections.defaultdict(list)
    record_index = collections.defaultdict(list)
    for index, row in batch.iterrows():
        text = row[text_column]
        language = row[language_column]
        if text and language:
            request_data[language].append(text)
            record_index[language].append(index)
        elif not text and language:
            request_data[language].append(' ')
            record_index[language].append(index)
        elif text and not language:
            request_data['en'].append(text)
            record_index['en'].append(index)
        else:
            request_data['en'].append(' ')
            record_index['en'].append(index)

    dct = collections.defaultdict(list)
    for language, request in request_data.items():
        re = client.batch_detect_key_phrases(TextList=request, LanguageCode=language)
        r = re.get('ResultList')
        dct[language] = r
    df_list = [None] * len(batch)
    for language, index_list in record_index.items():
        for i, index in enumerate(index_list):
            df_list[index] = dct[language][i]

    for i, row in batch.iterrows():
        text = row[text_column]
        row[output_column_name] = df_list[i] if text else None
        writer.write_row_dict(row)
