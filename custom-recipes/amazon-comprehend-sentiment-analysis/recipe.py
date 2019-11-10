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
language_column = get_recipe_config().get('language_column', None)
output_probabilities = get_recipe_config().get('output_probabilities', True)
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role('input-dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role('output-dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

output_schema = input_schema
predicted_sentiment_column = generate_unique('predicted_sentiment', input_columns_names)
output_schema.append({'name': predicted_sentiment_column, 'type': 'string'}) # 'predicted_sentiment' is consistent with other sentiment analysis plugins
if output_probabilities:
    predicted_probability_column = generate_unique('predicted_probability', input_columns_names)
    output_schema.append({'name': predicted_probability_column, 'type': 'double'}) # 'predicted_probability' is consistent with other sentiment analysis plugins
if should_output_raw_results:
    output_schema.append({"name": "raw_results", "type": "string"})
output_dataset.write_schema(output_schema)

with output_dataset.get_writer() as writer:
    for batch in input_dataset.iter_dataframes(chunksize=BATCH_SIZE):
        batch = batch.reset_index()
        output_rows = batch_detect_sentiment(batch, client, text_column, language_column)
        for row in output_rows:
            writer.write_row_dict(row)
