# -*- coding: utf-8 -*-
import logging
from typing import AnyStr, Dict

import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError

from io_utils import (
    generate_unique, safe_json_loads, ErrorHandlingEnum, OutputFormatEnum)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError)

BATCH_RESULT_KEY = "ResultList"
BATCH_ERROR_KEY = "ErrorList"
BATCH_INDEX_KEY = "Index"
BATCH_ERROR_MESSAGE_KEY = "ErrorMessage"
BATCH_ERROR_TYPE_KEY = "ErrorCode"

DEFAULT_AXIS_NUMBER = 1

ALL_ENTITY_TYPES = [
    'COMMERCIAL_ITEM', 'DATE', 'EVENT', 'LOCATION', 'ORGANIZATION',
    'OTHER', 'PERSON', 'QUANTITY', 'TITLE'
]

# ==============================================================================
# FUNCTION DEFINITION
# ==============================================================================

# TODO rewrite everything


def get_client(api_configuration_preset):
    client = boto3.client(
        service_name='comprehend',
        aws_access_key_id=api_configuration_preset.get('aws_access_key'),
        aws_secret_access_key=api_configuration_preset.get('aws_secret_key'),
        region_name=api_configuration_preset.get('aws_region'))
    logging.info("Credentials loaded")
    return client


def format_language_detection(
    row: Dict,
    response_column: AnyStr,
    column_prefix: AnyStr = "lang_detect_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    language_column = generate_unique(
        "language_code", row.keys(), column_prefix)
    row[language_column] = ''
    languages = response.get("Languages", [])
    if len(languages) > 0:
        row[language_column] = languages[0].get("LanguageCode", "")
    return row


def format_key_phrase_extraction(
    row: Dict,
    response_column: AnyStr,
    output_format: OutputFormatEnum = OutputFormatEnum.MULTIPLE_COLUMNS,
    num_key_phrases: int = 3,
    column_prefix: AnyStr = "keyphrase_api",
    error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> Dict:
    raw_response = row[response_column]
    response = safe_json_loads(raw_response, error_handling)
    if output_format == OutputFormatEnum.SINGLE_COLUMN:
        key_phrase_column = generate_unique(
            "keyphrase_list", row.keys(), column_prefix)
        row[key_phrase_column] = response.get("KeyPhrases", [])
    else:
        key_phrases = sorted(
            response.get("KeyPhrases", []), key=lambda x: x.get("Score"),
            reverse=True)
        for n in range(num_key_phrases):
            keyphrase_column = generate_unique(
                "keyphrase_" + str(n), row.keys(), column_prefix)
            score_column = generate_unique(
                "keyphrase_" + str(n) + "_score", row.keys(), column_prefix)
            if len(key_phrases) > n:
                row[keyphrase_column] = key_phrases[n].get("Text", '')
                row[score_column] = key_phrases[n].get("Score")
            else:
                row[keyphrase_column] = ''
                row[score_column] = None
    return row


def format_sentiment_results(raw_results):
    sentiment = raw_results.get('Sentiment').lower()
    output_row = dict()
    output_row["raw_results"] = raw_results
    output_row["predicted_sentiment"] = sentiment
    score = raw_results.get('SentimentScore', {}).get(sentiment.capitalize())
    output_row["predicted_probability"] = round(score, 2)
    return output_row


def format_language_results(raw_results):
    output_row = dict()
    output_row["raw_results"] = raw_results
    if len(raw_results.get('Languages')):
        language = raw_results.get('Languages')[0]
        output_row["detected_language"] = language.get('LanguageCode')
        output_row["probability"] = round(language.get('Score'), 2)
    else:
        output_row["detected_language"] = ''
        output_row["probability"] = ''
    return output_row


def format_keyphrases_results(raw_results):
    output_row = dict()
    output_row["raw_results"] = raw_results
    output_row["keyphrases"] = _distinct(
        [kp["Text"] for kp in raw_results.get("KeyPhrases", [])])
    return output_row


def format_entities_results(raw_results):
    output_row = dict()
    output_row["raw_results"] = raw_results
    output_row["entities"] = [_format_entity(
        e) for e in raw_results.get("Entities", [])]
    for t in ALL_ENTITY_TYPES:
        output_row[t] = _distinct(
            [e["text"] for e in output_row["entities"] if e["type"] == t])
    return output_row


def _format_entity(e):
    return {
        "type": e.get("Type"),
        "text": e.get("Text"),
        "score": e.get("Score"),
        "beginOffset": e.get("BeginOffset"),
        "endOffset": e.get("EndOffset"),
    }


def _distinct(l):
    return list(dict.fromkeys(l))
