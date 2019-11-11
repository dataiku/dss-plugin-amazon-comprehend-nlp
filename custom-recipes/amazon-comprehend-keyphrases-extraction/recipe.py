import logging
import time
import json
import dataiku
from dataiku.customrecipe import *
from dku_amazon_comprehend import *
from common import *

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[comprehend plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get('connectionInfo', {})
text_column = get_recipe_config().get('text_column')
language = get_recipe_config().get('language', 'en')
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role('input_dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]
keyphrases_column_name = generate_unique("keyphrases", input_columns_names)

output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

if text_column is None or len(text_column) == 0:
    raise ValueError("You must specify the input text column")
if text_column not in input_columns_names:
    raise ValueError("Column '{}' is not present in the input dataset".format(text_column))

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

input_df = input_dataset.get_dataframe()

@with_original_indices
def detect_key_phrases(text_list):
    logging.info("request: %d items / %d characters" % (len(text_list), sum([len(t) for t in text_list])))
    start = time.time()
    response = client.batch_detect_key_phrases(TextList=text_list, LanguageCode=language)
    logging.info("request took %.3fs" % (time.time() - start))
    return response


for batch in run_by_batch(detect_key_phrases, input_df, text_column, batch_size=BATCH_SIZE, parallelism=PARALLELISM):
    response, original_indices = batch
    if len(response.get('ErrorList', [])):
        logging.error(json.dumps(response.get('ErrorList')))
    for i, raw_result in enumerate(response.get('ResultList')):
        j = original_indices[i]
        output = format_keyphrases_results(raw_result)
        input_df.set_value(j, keyphrases_column_name, json.dumps(output['keyphrases']))
        if should_output_raw_results:
            input_df.set_value(j, 'raw_results', json.dumps(output['raw_results']))

output_dataset.write_with_schema(input_df)
