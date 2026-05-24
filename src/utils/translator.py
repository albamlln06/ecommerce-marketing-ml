from deep_translator import GoogleTranslator
import pandas as pd

def translate_column(
    df,
    column: str,
    target: str = "es",
    source: str = "auto",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Traduce una columna de un DataFrame.
    
    Parámetros:
    - df:      DataFrame de pandas
    - column:  Nombre de la columna a traducir
    - target:  Idioma destino (default: "es")
    - source:  Idioma origen (default: "auto")
    - inplace: Si True, sobreescribe la columna original
               Si False, crea una nueva columna '{column}_translated'
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el DataFrame")

    translator = GoogleTranslator(source=source, target=target)
    translated = df[column].astype(str).apply(translator.translate)

    if inplace:
        df[column] = translated
    else:
        df[f"{column}_translated"] = translated

    return df