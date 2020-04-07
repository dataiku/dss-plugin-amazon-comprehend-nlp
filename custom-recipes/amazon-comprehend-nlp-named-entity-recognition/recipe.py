# -*- coding: utf-8 -*-
import logging
import time
import json
import dataiku
from dataiku.customrecipe import *
from dku_aws_nlp import *
from api_calling_utils import *

# ==============================================================================
# SETUP
# ==============================================================================

logging.basicConfig(level=logging.INFO,
                    format='[comprehend plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get('connectionInfo', {})
text_column = get_recipe_config().get('text_column')
language = get_recipe_config().get('language', 'en')
output_format = get_recipe_config().get('output_format')
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_dataset_name = get_input_names_for_role('input_dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]
# Note: "entities" is consistent with other NER plugin
entities_column_name = generate_unique("entities", input_columns_names)

output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

if text_column is None or len(text_column) == 0:
    raise ValueError("You must specify the input text column")
if text_column not in input_columns_names:
    raise ValueError(
        "Column '{}' is not present in the input dataset".format(text_column))

client = get_client(connection_info)

# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()
response_column = generate_unique("raw_response", input_df.columns)
client = get_client(connection_info)


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
@fail_or_warn_on_row(error_handling=error_handling)
def call_api_named_entity_recognition(row, text_column, text_language=None):
    # TODO


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_named_entity_recognition,
    text_column=text_column, text_language=text_language,
    entity_sentiment=entity_sentiment, parallel_workers=parallel_workers)

output_df = output_df.apply(
    func=format_named_entity_recognition, axis=1,
    response_column=response_column, output_format=output_format,
    error_handling=error_handling)

output_dataset.write_with_schema(output_df)
