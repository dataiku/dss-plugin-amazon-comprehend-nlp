# -*- coding: utf-8 -*-
import logging
import json
from typing import Dict, AnyStr

from ratelimit import limits, RateLimitException
from retry import retry

import dataiku

from plugin_io_utils import (
    ErrorHandlingEnum, OutputFormatEnum,
    build_unique_column_names, validate_column_input)
from api_parallelizer import api_parallelizer
from dataiku.customrecipe import (
    get_recipe_config, get_input_names_for_role, get_output_names_for_role)
from cloud_api import (
    APPLY_AXIS, MedicalDetectionTypeEnum, get_client,
    format_medical_information_detection)


# ==============================================================================
# SETUP
# ==============================================================================

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
text_column = get_recipe_config().get('text_column')
medical_detection_type = MedicalDetectionTypeEnum[
    get_recipe_config().get('medical_detection_type')]
output_format = OutputFormatEnum[get_recipe_config().get('output_format')]
error_handling = ErrorHandlingEnum[get_recipe_config().get('error_handling')]

input_dataset_name = get_input_names_for_role('input_dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

validate_column_input(text_column, input_columns_names)

api_support_batch = False
column_prefix = "medical_api"
client = get_client(api_configuration_preset, "comprehendmedical")
input_df = input_dataset.get_dataframe()
api_column_names = build_unique_column_names(input_df, column_prefix)


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_medical(
    row: Dict, text_column: AnyStr,
    medical_detection_type: MedicalDetectionTypeEnum
) -> Dict:
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == '':
        return ''
    else:
        if medical_detection_type == MedicalDetectionTypeEnum.ENTITIES:
            responses = client.detect_entities_v2(Text=text)
        else:
            responses = client.detect_phi(Text=text)
        return json.dumps(responses)


output_df = api_parallelizer(
    input_df=input_df, api_call_function=call_api_medical,
    text_column=text_column, medical_detection_type=medical_detection_type,
    parallel_workers=parallel_workers, api_support_batch=api_support_batch,
    error_handling=error_handling, column_prefix=column_prefix)

logging.info("Formatting API results...")
output_df = output_df.apply(
   func=format_medical_information_detection, axis=APPLY_AXIS,
   response_column=api_column_names.response, output_format=output_format,
   medical_detection_type=medical_detection_type,
   column_prefix=column_prefix, error_handling=error_handling)
logging.info("Formatting API results: Done.")

output_dataset.write_with_schema(output_df)
