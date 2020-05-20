# -*- coding: utf-8 -*-
import logging
import json
from typing import Dict, List, Union, NamedTuple

import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError)


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_client(api_configuration_preset):
    client = boto3.client(
        service_name="comprehend",
        aws_access_key_id=api_configuration_preset.get("aws_access_key"),
        aws_secret_access_key=api_configuration_preset.get("aws_secret_key"),
        region_name=api_configuration_preset.get("aws_region"),
    )
    logging.info("Credentials loaded")
    return client


def batch_api_response_parser(batch: List[Dict], response: Union[Dict, List], api_column_names: NamedTuple) -> Dict:
    """
    Function to parse API results in the batch case.
    Needed for api_parallelizer.api_call_batch as each batch API needs specific response parsing.
    """
    results = response.get("ResultList", [])
    errors = response.get("ErrorList", [])
    for i in range(len(batch)):
        for k in api_column_names:
            batch[i][k] = ""
        result = [r for r in results if str(r.get("Index", "")) == str(i)]
        error = [r for r in errors if str(r.get("Index", "")) == str(i)]
        if len(result) != 0:
            # result must be json serializable
            batch[i][api_column_names.response] = json.dumps(result[0])
        if len(error) != 0:
            inner_error = error[0]
            logging.warning(str(inner_error))
            # custom for Azure edge case which is highly nested
            batch[i][api_column_names.error_message] = inner_error.get("ErrorMessage", "")
            batch[i][api_column_names.error_type] = inner_error.get("ErrorCode", "")
            batch[i][api_column_names.error_raw] = str(inner_error)
    return batch
