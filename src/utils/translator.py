import pandas as pd
from deep_translator import GoogleTranslator

def translate_reviews(df, field):

    traductor = GoogleTranslator(source='auto', target='en')

    df['english_review'] = df[field].apply(
        lambda x: traductor.translate(x) if isinstance(x, str) else x
    )
    return df