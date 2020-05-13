# -*- coding: utf-8 -*-
import json
from typing import List, Dict, AnyStr, Union

from retry import retry
from ratelimit import limits, RateLimitException

import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)

from plugin_io_utils import (
    ErrorHandlingEnum,
    validate_column_input,
    set_column_description,
)
from amazon_comprehend_api_client import (
    API_EXCEPTIONS,
    BATCH_RESULT_KEY,
    BATCH_ERROR_KEY,
    BATCH_INDEX_KEY,
    BATCH_ERROR_MESSAGE_KEY,
    BATCH_ERROR_TYPE_KEY,
    get_client,
)
from api_parallelizer import api_parallelizer
from amazon_comprehend_api_formatting import KeyPhraseExtractionAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
batch_size = api_configuration_preset.get("batch_size")
text_column = get_recipe_config().get("text_column")
text_language = get_recipe_config().get("language")
language_column = get_recipe_config().get("language_column")
num_key_phrases = int(get_recipe_config().get("num_key_phrases"))
error_handling = ErrorHandlingEnum[get_recipe_config().get("error_handling")]

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col["name"] for col in input_schema]

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

validate_column_input(text_column, input_columns_names)

batch_kwargs = {
    "api_support_batch": True,
    "batch_size": batch_size,
    "batch_result_key": BATCH_RESULT_KEY,
    "batch_error_key": BATCH_ERROR_KEY,
    "batch_index_key": BATCH_INDEX_KEY,
    "batch_error_message_key": BATCH_ERROR_MESSAGE_KEY,
    "batch_error_type_key": BATCH_ERROR_TYPE_KEY,
}
if text_language == "language_column":
    batch_kwargs = {"api_support_batch": False}
    validate_column_input(language_column, input_columns_names)

input_df = input_dataset.get_dataframe()
client = get_client(api_configuration_preset)
column_prefix = "keyphrase_api"


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_key_phrase_extraction(
    text_column: AnyStr,
    text_language: AnyStr,
    language_column: AnyStr = None,
    row: Dict = None,
    batch: List[Dict] = None,
) -> List[Union[Dict, AnyStr]]:
    if text_language == "language_column":
        # Cannot use batch as language may be different for each row
        text = row[text_column]
        language_code = row[language_column]
        empty_conditions = [
            not (isinstance(text, str)),
            not (isinstance(language_code, str)),
            str(text).strip() == "",
            str(language_code).strip() == "",
        ]
        if any(empty_conditions):
            return ""
        response = client.detect_key_phrases(Text=text, LanguageCode=language_code)
        return json.dumps(response)
    else:
        text_list = [str(r.get(text_column, "")).strip() for r in batch]
        responses = client.batch_detect_key_phrases(
            TextList=text_list, LanguageCode=text_language
        )
        return responses


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_key_phrase_extraction,
    api_exceptions=API_EXCEPTIONS,
    column_prefix=column_prefix,
    text_column=text_column,
    text_language=text_language,
    language_column=language_column,
    parallel_workers=parallel_workers,
    error_handling=error_handling,
    **batch_kwargs
)

api_formatter = KeyPhraseExtractionAPIFormatter(
    input_df=input_df,
    num_key_phrases=num_key_phrases,
    column_prefix=column_prefix,
    error_handling=error_handling,
)
output_df = api_formatter.format_df(df)

output_dataset.write_with_schema(output_df)
set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=api_formatter.column_description_dict,
)
