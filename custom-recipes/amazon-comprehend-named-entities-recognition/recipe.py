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
text_column = get_recipe_config().get('text_column')
language_column = get_recipe_config().get('language_column', 'en')
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

if text_column is None or len(text_column) == 0:
    raise ValueError("You must specify the input text column")

input_dataset_name = get_input_names_for_role('input-dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role('output-dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

if text_column not in input_columns_names:
    raise ValueError("Column '{}' is not present in the input dataset".format(text_column))

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

output_schema = input_schema
output_column_name = generate_unique("entities", input_columns_names) # Note: "entities" is consistent with other NER plugin
output_schema.append({'name': output_column_name, 'type': 'string'})
if should_output_raw_results:
    output_schema.append({"name": "raw_results", "type": "string"})
output_dataset.write_schema(output_schema)

with output_dataset.get_writer() as writer:
    for batch in input_dataset.iter_dataframes(chunksize=BATCH_SIZE):
        text_by_language, original_index_by_column = group_by_language(batch, text_column, language_column)
        results_per_language = collections.defaultdict(list)
        for language, request in text_by_language.items():
            response = client.batch_detect_entities(TextList=request, LanguageCode=language)
            results = response.get('ResultList')
            results_per_language[language] = results

        all_results = [None] * len(batch)
        for language, index_list in original_index_by_column.items():
            for i, index in enumerate(index_list):
                all_results[index] = results_per_language[language][i].get('Entities')

        for i, row in batch.iterrows():
            row[output_column_name] = all_results[i]
            writer.write_row_dict(row)
