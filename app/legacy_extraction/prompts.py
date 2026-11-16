"""Instructions pour extraire toutes les informations directement du PDF."""

PDF_EXTRACTION_PROMPT = """Analyse directement toutes les pages du PDF, y compris texte, tableaux, graphiques, titres et mise en page.

Extrais uniquement les informations lisibles et supportees par le document. N'invente rien, n'utilise aucune connaissance externe et retourne null ou [] si une information est absente, ambigue ou illisible.

Pour chaque information, donne la page, un source_text exact et tres court, une confiance entre 0 et 1 et la methode. source_text est le plus petit extrait suffisant (expression ou ligne contenant directement la valeur), jamais un paragraphe ni un titre de section repete. Ne reformule pas.

Si "FICHE MENSUELLE" est explicitement visible, utilise MONTHLY_FACT_SHEET avec source_method explicit, jamais FINANCIAL_REPORT. Sinon, document_type peut etre infere. Ne fabrique jamais document_title. Ne confonds pas document_date avec une date de reunion, decision, valorisation, finance ou mandat.

Extrais la section INFORMATIONS CLES dans fund_information. Extrais chaque regle d'orientation de placement dans investment_policy en conservant toujours raw_text; minimum, maximum et percentage restent null si leur sens n'est pas explicite. Extrais dans historical_performance les valeurs historiques lisibles des graphiques ou tableaux; n'invente aucune valeur visuelle incertaine.

Preserve chaque valeur financiere dans raw_value telle qu'elle apparait. Pour une metrique, label est un libelle categoriel court, raw_value contient seulement la valeur brute, et source_text contient le plus petit extrait qui relie directement les deux; ne repete pas une phrase longue dans ces trois champs.

Classe BOND_ALLOCATION toute valeur explicitement rattachee par le titre de section ou le contexte immediat a une repartition obligataire: maturite obligataire, emprunts nationaux, emprunts societes ou repartition des emprunts obligataires. Une allocation sans contexte obligataire explicite reste ALLOCATION. Ne fais aucune conversion financiere implicite. numeric_value doit rester null si le nombre est ambigu; unit et currency sont distincts et MD n'est pas automatiquement une devise. Distingue date, maturity et percentage.

Retourne toutes les occurrences pertinentes de tout le document et respecte exactement le schema JSON fourni."""


def pdf_chunk_prompt(start_page: int, end_page: int) -> str:
    """Ajoute le contexte necessaire a l'extraction d'un bloc de pages."""
    return f"""{PDF_EXTRACTION_PROMPT}

Tu analyses seulement un extrait de pages d'un document plus large.
Cet extrait correspond aux pages {start_page} a {end_page} du PDF original.
Extrais uniquement ce qui est explicitement present dans ces pages.
Ne conclus pas qu'une information est absente du document complet.
Pour page_number, utilise si possible le numero de page du PDF original."""
