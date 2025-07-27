def frage_benutzer(msg):
    return input(f"{msg} [j/n]: ").strip().lower() == 'j'
