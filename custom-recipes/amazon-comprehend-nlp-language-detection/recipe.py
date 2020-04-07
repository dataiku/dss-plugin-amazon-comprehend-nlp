# -*- coding: utf-8 -*-
import logging
import json

from ratelimit import limits, RateLimitException
from retry import retry

import dataiku
from api_calling_utils import (
    generate_unique, fail_or_warn_on_row, api_parallelizer
)
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role, get_output_names_for_role
)
from dku_aws_nlp import get_client


# ==============================================================================
# SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[Amazon Comprehend NLP plugin] %(levelname)s - %(message)s'
)

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
batch_size = api_configuration_preset.get("batch_size")
text_column = get_recipe_config().get('text_column')
error_handling = get_recipe_config().get('error_handling')

input_dataset_name = get_input_names_for_role('input_dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

if text_column is None or len(text_column) == 0:
    raise ValueError("You must specify the input text column.")
if text_column not in input_columns_names:
    raise ValueError(
        "Column '{}' is not present in the input dataset.".format(text_column)
    )


# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()
response_column = generate_unique("raw_response", input_df.columns)
client = get_client(api_configuration_preset)


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_language_detection(row, text_column):
    response_key = generate_unique("raw_response", row[0].keys())
    error_message_key = generate_unique("error_message", row[0].keys())
    error_type_key = generate_unique("error_type", row[0].keys())
    error_raw_key = generate_unique("error_raw", row[0].keys())
    new_keys = [
        response_key, error_message_key,
        error_type_key, error_raw_key
    ]
    text_list = [r[text_column] for r in row]
    response = client.batch_detect_dominant_language(TextList=text_list)
    print(response)
    for i in range(len(text_list)):
        for k in new_keys:
            row[i][k] = ''
        row[i][response_column] = json.dumps([
            d.get("Languages", [])
            for d in response["ResultList"] if d["Index"] == i
        ][0])
    return row


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_language_detection,
    text_column=text_column, parallel_workers=parallel_workers,
    api_support_batch=True, batch_size=batch_size)

# output_df = output_df.apply(
#    func=format_named_entity_recognition, axis=1,
#    response_column=response_column, output_format=output_format,
#    error_handling=error_handling)

output_dataset.write_with_schema(output_df)
