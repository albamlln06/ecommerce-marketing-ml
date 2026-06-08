import os
import gdown
import pandas as pd
import tempfile


READERS = {
    "csv":     lambda p: pd.read_csv(p),
    "json":    lambda p: pd.read_json(p),
    "excel":   lambda p: pd.read_excel(p),
    "parquet": lambda p: pd.read_parquet(p),
}

WRITERS = {
    "parquet": lambda df, p: df.to_parquet(p, index=False, engine="pyarrow"),
    "csv":     lambda df, p: df.to_csv(p, index=False),
    "json":    lambda df, p: df.to_json(p, orient="records", indent=2),
    "excel":   lambda df, p: df.to_excel(p, index=False),
}


def fetch_file(file_url: str, input_format: str = "csv") -> pd.DataFrame:
    """
    Descarga un único archivo de Google Drive y lo retorna como DataFrame.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, f"file.{input_format}")
        
        gdown.download(file_url, tmp_path, quiet=False)
        df = READERS[input_format](tmp_path)
        print(f"✅ Archivo cargado → {df.shape[0]} filas, {df.shape[1]} columnas")

    return df


def convert_files(
    input_path: str = "data/",
    output_path: str = "data_converted/",
    input_format: str = "csv",
    output_format: str = "parquet",
) -> list[str]:
    """
    Convierte todos los archivos de input_path al formato deseado
    y los guarda en output_path.
    Retorna lista de rutas de archivos convertidos.
    """
    if input_format not in READERS:
        raise ValueError(f"Formato de entrada no soportado: {input_format}. Opciones: {list(READERS)}")
    if output_format not in WRITERS:
        raise ValueError(f"Formato de salida no soportado: {output_format}. Opciones: {list(WRITERS)}")

    os.makedirs(output_path, exist_ok=True)
    read  = READERS[input_format]
    write = WRITERS[output_format]

    converted = []
    files = [f for f in os.listdir(input_path) if f.endswith(f".{input_format}")]

    if not files:
        print(f"No se encontraron archivos .{input_format} en '{input_path}'")
        return converted

    for filename in files:
        src  = os.path.join(input_path, filename)
        dest = os.path.join(output_path, filename.replace(f".{input_format}", f".{output_format}"))
        try:
            df = read(src)
            write(df, dest)
            converted.append(dest)
            print(f"{filename} in {os.path.basename(dest)}")
        except Exception as e:
            print(f"Error en {filename}: {e}")

    print(f"Archivos transformados en '{output_path}'")
    return converted