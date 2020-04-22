# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, AnyStr

from retry import retry
from ratelimit import limits, RateLimitException

import dataiku

from plugin_io_utils import (
    ErrorHandlingEnum, build_unique_column_names, validate_column_input)
from api_parallelizer import api_parallelizer
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role, get_output_names_for_role)
from cloud_api import (
    API_SUPPORT_BATCH, APPLY_AXIS,
    get_client, format_language_detection)


# ==============================================================================
# SETUP
# ==============================================================================

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

validate_column_input(text_column, input_columns_names)
input_df = input_dataset.get_dataframe()
client = get_client(api_configuration_preset)
column_prefix = "lang_detect_api"
api_column_names = build_unique_column_names(input_df, column_prefix)


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_language_detection(
    batch: List[Dict], text_column: AnyStr
) -> List[Dict]:
    text_list = [str(r.get(text_column, '')).strip() for r in batch]
    responses = client.batch_detect_dominant_language(TextList=text_list)
    return responses


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_language_detection,
    text_column=text_column, parallel_workers=parallel_workers,
    api_support_batch=API_SUPPORT_BATCH, batch_size=batch_size,
    error_handling=error_handling, column_prefix=column_prefix)

logging.info("Formatting API results...")
output_df = output_df.apply(
   func=format_language_detection, axis=APPLY_AXIS,
   response_column=api_column_names.response, error_handling=error_handling,
   column_prefix=column_prefix)
logging.info("Formatting API results: Done.")

output_dataset.write_with_schema(output_df)
