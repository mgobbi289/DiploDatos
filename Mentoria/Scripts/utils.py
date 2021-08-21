from re import sub
from unidecode import unidecode

def cleaning(title):
    """
    Aplica las operaciones de limpieza a un título de una publicación.
    """
    # Unidecode: Convierte string de Unicode a ASCII.
    title = unidecode(title)
    # Pasamos a Minúsculas.
    title = title.lower()
    # Eliminamos Números.
    title = sub(r'[0-9]+', '', title)
    # Reemplazamos Contracciones.
    title = sub(r'c/u', 'cada uno', title)
    title = sub(r'c/', 'con', title)
    title = sub(r'p/', 'para', title)
    # Limpiamos Símbolos.
    title = sub('[^a-zA-Z ]', '', title)
    # Retornamos el título de la publicación procesado.
    return title

