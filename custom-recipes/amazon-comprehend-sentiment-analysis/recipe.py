import logging
import collections
import dataiku
from dataiku.customrecipe import *
from dku_amazon_comprehend import *

BATCH_SIZE = 10

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='aws-machine-learning plugin %(levelname)s - %(message)s')

#==============================================================================
# SETUP
#==============================================================================

input_dataset_name = get_input_names_for_role('input-dataset')[0]
output_dataset_name = get_output_names_for_role('output-dataset')[0]

connection_info = get_recipe_config().get('connectionInfo', {})
text_column = get_recipe_config().get('text_column', None)
language_column = get_recipe_config().get('language_column', None)
output_probabilities = get_recipe_config().get('output_probabilities', True)

input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col['name'] for col in input_schema]

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
output_dataset.write_schema(output_schema)

writer = output_dataset.get_writer()

for batch in input_dataset.iter_dataframes(chunksize=BATCH_SIZE):
    batch = batch.reset_index()
    text_by_language, original_indices_by_language = group_by_language(batch, text_column, language_column)
    results_per_language = collections.defaultdict(list)
    for language, request in text_by_language.items():
        re = client.batch_detect_sentiment(TextList=request, LanguageCode=language)
        results_per_language[language] = re.get('ResultList')
    result_per_row = [None] * len(batch)
    for language, original_indices in original_indices_by_language.items():
        for i, original_index in enumerate(original_indices):
            result_per_row[original_index] = results_per_language[language][i]

    for original_index, row in batch.iterrows():
        text = row[text_column]
        result = result_per_row[original_index]
        if text:
            sentiment = result.get('Sentiment').lower()
            row[predicted_sentiment_column] = sentiment
            if output_probabilities:
                row[predicted_probability_column] = result.get('SentimentScore',{}).get(sentiment.capitalize())
        writer.write_row_dict(row)
