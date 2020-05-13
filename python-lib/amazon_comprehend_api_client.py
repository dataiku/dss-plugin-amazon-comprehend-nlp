# -*- coding: utf-8 -*-
import logging

import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError)

API_SUPPORT_BATCH = True
BATCH_RESULT_KEY = "ResultList"
BATCH_ERROR_KEY = "ErrorList"
BATCH_INDEX_KEY = "Index"
BATCH_ERROR_MESSAGE_KEY = "ErrorMessage"
BATCH_ERROR_TYPE_KEY = "ErrorCode"


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
