from SPARQLWrapper import SPARQLWrapper, JSON
import json
from tqdm import tqdm


dbpedia_endpoint_url = "https://dbpedia.org/sparql"
sparql = SPARQLWrapper(dbpedia_endpoint_url)
sparql.setReturnFormat(JSON)
PREDICATE_WHITELIST = [
    "http://dbpedia.org/ontology/director",
    "http://dbpedia.org/ontology/starring",
    "http://dbpedia.org/ontology/producer",
    "http://dbpedia.org/ontology/writer",
    "http://dbpedia.org/ontology/musicComposer",
    "http://dbpedia.org/ontology/cinematography",
    "http://dbpedia.org/ontology/editing",
    "http://dbpedia.org/ontology/distributor",
    "http://dbpedia.org/ontology/country",
    "http://dbpedia.org/ontology/language",
    "http://dbpedia.org/ontology/abstract"  # abstract ovvero il testo
]
ABSTRACT_URI = "http://dbpedia.org/ontology/abstract"

class JSONRetrieve:
    def __init__(self,test_enable,input_filename,output_filename):
        self.test_enable = test_enable
        self.input_filename = input_filename
        self.output_filename = output_filename


    # conserviamo dell'abstract solo il primo paragrafo
    def get_first_paragraph(self,full_text):
        if not full_text: return ""
        paragraphs = full_text.split('\n')
        for p in paragraphs:
            if p.strip(): return p.strip()
        return ""


    def get_short_name(self,uri_string):
        """Converte un URI completo nel formato con prefisso (es. dbo:, dbr:)"""
        if not isinstance(uri_string, str):
            return ""

        # Dizionario per mappare gli URI ai prefissi
        uri_namespace_map = {
            "http://dbpedia.org/resource/": "dbr:",
            "http://dbpedia.org/ontology/": "dbo:",
            "http://dbpedia.org/property/": "dbp:",
        }

        # Cerca il namespace corrispondente
        for namespace_uri, prefix in uri_namespace_map.items():
            if uri_string.startswith(namespace_uri):
                local_name = uri_string[len(namespace_uri):]
                return f"{prefix}{local_name}"

        # Fallback: se non trova un namespace noto, restituisce solo il nome locale
        if uri_string.startswith("http://"):
            return uri_string.split('/')[-1]

        return uri_string

    def compute(self):
        with open(self.input_filename, "r", encoding="utf-8") as f:
            film_uris = [line.strip() for line in f if line.strip()]

        if self.test_enable:
            film_uris = film_uris[:500] # TODO da rimuovere, solo per test
        print(f"Letti {len(film_uris)} URI di film da '{self.input_filename}'.")

        # whitelist dei predicati. Da capire se sono troppi
        final_dataset = []

        for i, film_uri in enumerate(tqdm(film_uris, desc="Creando il dataset finale")):
            # limita i predicati a quelli nella whitelist
            values_clause = "VALUES ?p { " + " ".join([f"<{p}>" for p in PREDICATE_WHITELIST]) + " }"

            # query per ottenere tutti i predicati "?p" e oggetti "?o" per il film "film_uri"
            query = f"""
                SELECT ?p ?o
                WHERE {{
                  <{film_uri}> ?p ?o .
                  {values_clause}
                  FILTER (!isLiteral(?o) || lang(?o) = "en")
                }}
            """

            try:
                # esecuzione della query
                sparql.setQuery(query)
                results = sparql.query().convert()["results"]["bindings"]

                if not results: continue

                film_abstract_text = ""
                film_triples = []

                # si processano tutti i risultati ottenuti
                for res in results:
                    predicate_uri = res['p']['value']
                    obj_info = res['o']

                    # se è l'abstract, lo salviamo a parte
                    if predicate_uri == ABSTRACT_URI:
                        film_abstract_text = obj_info['value']
                    # altrimenti è una tripla
                    else:
                        film_triples.append({
                            "subject": self.get_short_name(film_uri),
                            "predicate": self.get_short_name(predicate_uri),
                            "object": self.get_short_name(obj_info['value'])
                        })


                # TODO da capire se mantenere o no
                first_paragraph = self.get_first_paragraph(film_abstract_text)

                # se abbiamo trovato l'abstract e almeno una tripla, è una valida entry del dataset
                if first_paragraph and film_triples:
                    final_dataset.append({
                        "uri": film_uri,
                        "text": first_paragraph,
                        "triples": film_triples
                    })

            except Exception as e:
                print(f"\nErrore durante il recupero di {film_uri}: {e}")

        # salvataggio in JSON
        with open(self.output_filename, "w", encoding="utf-8") as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)

        print(f"\nProcesso completato. Dataset finale salvato in '{self.output_filename}'.")
        print(f"Creati {len(final_dataset)} esempi validi (testo + triple).")