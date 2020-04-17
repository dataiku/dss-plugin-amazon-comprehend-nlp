# -*- coding: utf-8 -*-
import logging

from ratelimit import limits, RateLimitException
from retry import retry

import dataiku
from api_calling_utils import initialize_api_column_names, api_parallelizer
from param_enums import ErrorHandlingEnum
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
error_handling = ErrorHandlingEnum[get_recipe_config().get('error_handling')]

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
        "Column '{}' is not present in the input dataset.".format(text_column))


# ==============================================================================
# RUN
# ==============================================================================

input_df = input_dataset.get_dataframe()
client = get_client(api_configuration_preset)
column_prefix = "lang_detect_api"
api_column_dict = initialize_api_column_names(input_df, column_prefix)


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_language_detection(batch, text_column):
    text_list = [str(r.get(text_column, '')).strip() for r in batch]
    responses = client.batch_detect_dominant_language(TextList=text_list)
    return responses


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_language_detection,
    text_column=text_column, parallel_workers=parallel_workers,
    api_support_batch=True, batch_size=batch_size,
    error_handling=error_handling, column_prefix=column_prefix,
    batch_result_key="ResultList", batch_error_key="ErrorList",
    batch_index_key="Index", batch_error_message_key="ErrorMessage",
    batch_error_type_key="ErrorCode",
)

# output_df = output_df.apply(
#    func=format_named_entity_recognition, axis=1,
#    response_column=response_column, output_format=output_format,
#    error_handling=error_handling)

output_dataset.write_with_schema(output_df)
