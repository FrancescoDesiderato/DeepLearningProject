from SPARQLWrapper import SPARQLWrapper, JSON
import time

# connessione all'endpoint SPARQL di DBpedia
dbpedia_endpoint_url = "https://dbpedia.org/sparql"
sparql = SPARQLWrapper(dbpedia_endpoint_url)
sparql.setReturnFormat(JSON)

# Impostiamo la dimensione della pagina
page_size = 5000  # Possiamo mantenere una page size alta
all_film_uris = []

# per tenere traccia dell'ultimo URI visto
last_uri = None

print("Inizio estrazione degli URI...")

while True:
    # FILTER assicura che otteniamo solo URI maggiori dell'ultimo visto
    if last_uri:
        # se esiste un uri precedente, si usa per il filtro
        filter_clause = f'FILTER (?film > <{last_uri}>)'
    else:
        filter_clause = ''

    # query da passare all'endpoint
    query = f"""
        SELECT DISTINCT ?film
        WHERE {{
          ?film a dbo:Film .
          {filter_clause}
        }}
        ORDER BY ?film
        LIMIT {page_size}
    """

    try:
        # esecuzione della query
        sparql.setQuery(query)
        results = sparql.query().convert()

        bindings = results["results"]["bindings"]

        # Se la pagina è vuota, abbiamo finito
        if not bindings:
            print("Nessun altro risultato trovato. Estrazione completata.")
            break


        # estrazione degli URI dalla pagina corrente
        page_uris = [item['film']['value'] for item in bindings]
        all_film_uris.extend(page_uris)

        # Aggiorna l'ultimo URI visto con l'ultimo elemento di questa pagina
        last_uri = page_uris[-1]

        print(f"Trovati {len(page_uris)} URI. Totale finora: {len(all_film_uris)}. Ultimo URI: {last_uri}")

        # Pausa
        time.sleep(1)

    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        time.sleep(10)

print(f"\nEstrazione terminata. Numero totale di URI di film raccolti: {len(all_film_uris)}")

# Salvataggio degli URI in un file di testo
output_filename = "film_uris.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    for uri in all_film_uris:
        f.write(uri + "\n")

