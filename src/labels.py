"""Label definitions for zero-shot classification."""

from typing import Dict, List


LABEL_SETS = {
    "ag_news": {
        "name_only": {
            0: ["world"],
            1: ["sports"],
            2: ["business"],
            3: ["science and technology"],
        },
        "description": {
            0: ["This text is about international events, global politics, diplomacy, conflicts, or world affairs."],
            1: ["This text is about sports, matches, teams, athletes, tournaments, or competitions."],
            2: ["This text is about business, markets, finance, companies, trade, or the economy."],
            3: ["This text is about science, technology, computers, innovation, research, or digital products."],
        },
    },
    
    "dbpedia_14": {
        "name_only": {
            0: ["company"],
            1: ["educational institution"],
            2: ["artist"],
            3: ["athlete"],
            4: ["office holder"],
            5: ["mean of transportation"],
            6: ["building"],
            7: ["natural place"],
            8: ["village"],
            9: ["animal"],
            10: ["plant"],
            11: ["album"],
            12: ["film"],
            13: ["written work"],
        },
        "description": {
            0: ["This text describes a company, corporation, or business organization."],
            1: ["This text describes an educational institution, school, university, or academy."],
            2: ["This text describes an artist, musician, painter, sculptor, or creative person."],
            3: ["This text describes an athlete, sports person, or competitor."],
            4: ["This text describes an office holder, politician, government official, or elected representative."],
            5: ["This text describes a means of transportation, vehicle, transit system, or mode of travel."],
            6: ["This text describes a building, structure, architectural work, or constructed facility."],
            7: ["This text describes a natural place, geographical feature, landscape, or natural formation."],
            8: ["This text describes a village, town, small settlement, or community."],
            9: ["This text describes an animal species, creature, or living organism."],
            10: ["This text describes a plant species, vegetation, flora, or botanical entity."],
            11: ["This text describes a music album, record, collection of songs, or musical release."],
            12: ["This text describes a film, movie, cinema production, or motion picture."],
            13: ["This text describes a written work, book, literary piece, publication, or document."],
        },
    },
    
    "yahoo_answers_topics": {
        "name_only": {
            0: ["society and culture"],
            1: ["science and mathematics"],
            2: ["health"],
            3: ["education and reference"],
            4: ["computers and internet"],
            5: ["sports"],
            6: ["business and finance"],
            7: ["entertainment and music"],
            8: ["family and relationships"],
            9: ["politics and government"],
        },
        "description": {
            0: ["This question is about society, culture, social issues, traditions, or cultural practices."],
            1: ["This question is about science, mathematics, physics, chemistry, biology, or scientific concepts."],
            2: ["This question is about health, medicine, diseases, wellness, nutrition, or medical conditions."],
            3: ["This question is about education, learning, schools, teaching, references, or academic topics."],
            4: ["This question is about computers, internet, technology, software, hardware, or digital topics."],
            5: ["This question is about sports, athletics, games, competitions, or physical activities."],
            6: ["This question is about business, finance, economy, investing, or commercial activities."],
            7: ["This question is about entertainment, music, movies, celebrities, arts, or leisure activities."],
            8: ["This question is about family, relationships, marriage, parenting, or interpersonal connections."],
            9: ["This question is about politics, government, laws, policies, or political issues."],
        },
    },
    
    "banking77": {
        "name_only": {
            0: ["activate my card"],
            1: ["age limit"],
            2: ["apple pay or google pay"],
            3: ["atm support"],
            4: ["automatic top up"],
            5: ["balance not updated after bank transfer"],
            6: ["balance not updated after cheque or cash deposit"],
            7: ["beneficiary not allowed"],
            8: ["cancel transfer"],
            9: ["card about to expire"],
            10: ["card acceptance"],
            11: ["card arrival"],
            12: ["card delivery estimate"],
            13: ["card linking"],
            14: ["card not working"],
            15: ["card payment fee charged"],
            16: ["card payment not recognised"],
            17: ["card payment wrong exchange rate"],
            18: ["card swallowed"],
            19: ["cash withdrawal charge"],
            20: ["cash withdrawal not recognised"],
            21: ["change pin"],
            22: ["compromised card"],
            23: ["contactless not working"],
            24: ["country support"],
            25: ["declined card payment"],
            26: ["declined cash withdrawal"],
            27: ["declined transfer"],
            28: ["direct debit payment not recognised"],
            29: ["disposable card limits"],
            30: ["edit personal details"],
            31: ["exchange charge"],
            32: ["exchange rate"],
            33: ["exchange via app"],
            34: ["extra charge on statement"],
            35: ["failed transfer"],
            36: ["fiat currency support"],
            37: ["get disposable virtual card"],
            38: ["get physical card"],
            39: ["getting spare card"],
            40: ["getting virtual card"],
            41: ["lost or stolen card"],
            42: ["lost or stolen phone"],
            43: ["order physical card"],
            44: ["passcode forgotten"],
            45: ["pending card payment"],
            46: ["pending cash withdrawal"],
            47: ["pending top up"],
            48: ["pending transfer"],
            49: ["pin blocked"],
            50: ["receiving money"],
            51: ["refund not showing up"],
            52: ["request refund"],
            53: ["reverted card payment"],
            54: ["supported cards and currencies"],
            55: ["terminate account"],
            56: ["top up by bank transfer charge"],
            57: ["top up by card charge"],
            58: ["top up by cash or cheque"],
            59: ["top up failed"],
            60: ["top up limits"],
            61: ["top up reverted"],
            62: ["topping up by card"],
            63: ["transaction charged twice"],
            64: ["transfer fee charged"],
            65: ["transfer into account"],
            66: ["transfer not received by recipient"],
            67: ["transfer timing"],
            68: ["unable to verify identity"],
            69: ["verify my identity"],
            70: ["verify source of funds"],
            71: ["verify top up"],
            72: ["virtual card not working"],
            73: ["visa or mastercard"],
            74: ["why verify identity"],
            75: ["wrong amount of cash received"],
            76: ["wrong exchange rate for cash withdrawal"],
        },
        "description": {
            0: ["The user wants to activate their card or asking how to activate it."],
            1: ["The user is asking about age limits or age requirements for services."],
            2: ["The user has a question about Apple Pay or Google Pay integration."],
            3: ["The user needs information about ATM support or ATM locations."],
            4: ["The user wants to set up automatic top up or asking about this feature."],
            5: ["The user's balance is not updated after a bank transfer."],
            6: ["The user's balance is not updated after depositing a cheque or cash."],
            7: ["The user cannot add a beneficiary or the beneficiary is not allowed."],
            8: ["The user wants to cancel a transfer or stop a pending transfer."],
            9: ["The user's card is about to expire and they need information."],
            10: ["The user has questions about where their card is accepted."],
            11: ["The user is asking about when their card will arrive."],
            12: ["The user wants an estimate for card delivery time."],
            13: ["The user has issues linking their card or wants to link cards."],
            14: ["The user's card is not working properly."],
            15: ["The user was charged a fee for a card payment."],
            16: ["The user's card payment was not recognised or recorded."],
            17: ["The user was charged wrong exchange rate for card payment."],
            18: ["The user's card was swallowed by an ATM."],
            19: ["The user was charged for cash withdrawal."],
            20: ["The user's cash withdrawal was not recognised."],
            21: ["The user wants to change their PIN code."],
            22: ["The user believes their card is compromised or stolen."],
            23: ["The user's contactless payment is not working."],
            24: ["The user is asking about country support or international usage."],
            25: ["The user's card payment was declined."],
            26: ["The user's cash withdrawal was declined."],
            27: ["The user's transfer was declined."],
            28: ["The user's direct debit payment was not recognised."],
            29: ["The user is asking about disposable card limits."],
            30: ["The user wants to edit their personal details."],
            31: ["The user has questions about exchange charges or fees."],
            32: ["The user is asking about exchange rates."],
            33: ["The user wants to exchange currency via the app."],
            34: ["The user sees an extra charge on their statement."],
            35: ["The user's transfer has failed."],
            36: ["The user is asking about fiat currency support."],
            37: ["The user wants to get a disposable virtual card."],
            38: ["The user wants to get a physical card."],
            39: ["The user wants to get a spare card."],
            40: ["The user wants to get a virtual card."],
            41: ["The user has lost their card or it was stolen."],
            42: ["The user has lost their phone or it was stolen."],
            43: ["The user wants to order a physical card."],
            44: ["The user has forgotten their passcode."],
            45: ["The user's card payment is pending."],
            46: ["The user's cash withdrawal is pending."],
            47: ["The user's top up is pending."],
            48: ["The user's transfer is pending."],
            49: ["The user's PIN is blocked."],
            50: ["The user has questions about receiving money."],
            51: ["The user's refund is not showing up."],
            52: ["The user wants to request a refund."],
            53: ["The user's card payment was reverted."],
            54: ["The user is asking about supported cards and currencies."],
            55: ["The user wants to terminate their account."],
            56: ["The user was charged for top up by bank transfer."],
            57: ["The user was charged for top up by card."],
            58: ["The user wants to top up by cash or cheque."],
            59: ["The user's top up has failed."],
            60: ["The user is asking about top up limits."],
            61: ["The user's top up was reverted."],
            62: ["The user wants information about topping up by card."],
            63: ["The user was charged twice for the same transaction."],
            64: ["The user was charged a transfer fee."],
            65: ["The user is asking about transfer into account."],
            66: ["The user's transfer was not received by the recipient."],
            67: ["The user is asking about transfer timing or how long transfers take."],
            68: ["The user is unable to verify their identity."],
            69: ["The user wants to verify their identity."],
            70: ["The user needs to verify their source of funds."],
            71: ["The user needs to verify their top up."],
            72: ["The user's virtual card is not working."],
            73: ["The user is asking about Visa or Mastercard."],
            74: ["The user wants to know why they need to verify identity."],
            75: ["The user received the wrong amount of cash."],
            76: ["The user was charged wrong exchange rate for cash withdrawal."],
        },
    },
    
    # Twitter Financial Sentiment
    "zeroshot/twitter-financial-news-sentiment": {
        "name_only": {
            0: ["bearish"],
            1: ["bullish"],
            2: ["neutral"],
        },
        "description": {
            0: ["This text expresses bearish sentiment, negative outlook, pessimism, or expectation of price decline in financial markets."],
            1: ["This text expresses bullish sentiment, positive outlook, optimism, or expectation of price increase in financial markets."],
            2: ["This text expresses neutral sentiment, objective reporting, or balanced view without clear positive or negative bias in financial markets."],
        },
    },
    
    # 20 Newsgroups
    "SetFit/20_newsgroups": {
        "name_only": {
            0: ["alt.atheism"],
            1: ["comp.graphics"],
            2: ["comp.os.ms-windows.misc"],
            3: ["comp.sys.ibm.pc.hardware"],
            4: ["comp.sys.mac.hardware"],
            5: ["comp.windows.x"],
            6: ["misc.forsale"],
            7: ["rec.autos"],
            8: ["rec.motorcycles"],
            9: ["rec.sport.baseball"],
            10: ["rec.sport.hockey"],
            11: ["sci.crypt"],
            12: ["sci.electronics"],
            13: ["sci.med"],
            14: ["sci.space"],
            15: ["soc.religion.christian"],
            16: ["talk.politics.guns"],
            17: ["talk.politics.mideast"],
            18: ["talk.politics.misc"],
            19: ["talk.religion.misc"],
        },
        "description": {
            0: ["This text discusses atheism, religious skepticism, secular humanism, or non-religious philosophy."],
            1: ["This text discusses computer graphics, image processing, visualization, rendering, or graphical software."],
            2: ["This text discusses Microsoft Windows operating system issues, tips, or questions."],
            3: ["This text discusses IBM PC hardware, components, upgrades, or technical specifications."],
            4: ["This text discusses Apple Macintosh hardware, components, or technical specifications."],
            5: ["This text discusses X Window System, Unix graphical interface, or related software."],
            6: ["This text is a for-sale advertisement, marketplace listing, or commercial offer."],
            7: ["This text discusses automobiles, cars, driving, automotive technology, or vehicle maintenance."],
            8: ["This text discusses motorcycles, bikes, riding, or motorcycle maintenance."],
            9: ["This text discusses baseball, MLB, baseball players, games, or statistics."],
            10: ["This text discusses hockey, NHL, hockey players, games, or ice hockey."],
            11: ["This text discusses cryptography, encryption, security algorithms, or cryptographic systems."],
            12: ["This text discusses electronics, circuits, electronic components, or electrical engineering."],
            13: ["This text discusses medicine, medical conditions, healthcare, or medical research."],
            14: ["This text discusses space, astronomy, space exploration, NASA, or astrophysics."],
            15: ["This text discusses Christianity, Christian faith, Bible, or Christian theology."],
            16: ["This text discusses gun politics, firearms, gun rights, or gun control debates."],
            17: ["This text discusses Middle East politics, conflicts, or geopolitical issues in that region."],
            18: ["This text discusses general political topics, political debates, or miscellaneous political issues."],
            19: ["This text discusses general religious topics, interfaith dialogue, or miscellaneous religious matters."],
        },
    },
    
    # IMDB Movie Reviews - Binary sentiment
    "imdb": {
        "name_only": {
            0: ["negative"],
            1: ["positive"],
        },
        "description": {
            0: ["This text expresses negative sentiment, criticism, disappointment, or unfavorable opinion about a movie."],
            1: ["This text expresses positive sentiment, praise, enjoyment, or favorable opinion about a movie."],
        },
    },
    
    # SST-2 (Stanford Sentiment Treebank) - Binary sentiment
    "sst2": {
        "name_only": {
            0: ["negative"],
            1: ["positive"],
        },
        "description": {
            0: ["This text expresses negative sentiment, criticism, disappointment, or unfavorable opinion."],
            1: ["This text expresses positive sentiment, praise, satisfaction, or favorable opinion."],
        },
    },
    
    # GoEmotions - 28 emotion categories (27 + neutral)
    "go_emotions": {
        "name_only": {
            0: ["admiration"],
            1: ["amusement"],
            2: ["anger"],
            3: ["annoyance"],
            4: ["approval"],
            5: ["caring"],
            6: ["confusion"],
            7: ["curiosity"],
            8: ["desire"],
            9: ["disappointment"],
            10: ["disapproval"],
            11: ["disgust"],
            12: ["embarrassment"],
            13: ["excitement"],
            14: ["fear"],
            15: ["gratitude"],
            16: ["grief"],
            17: ["joy"],
            18: ["love"],
            19: ["nervousness"],
            20: ["optimism"],
            21: ["pride"],
            22: ["realization"],
            23: ["relief"],
            24: ["remorse"],
            25: ["sadness"],
            26: ["surprise"],
            27: ["neutral"],
        },
        "description": {
            0: ["This text expresses admiration, respect, appreciation, or positive regard for someone or something."],
            1: ["This text expresses amusement, humor, laughter, or finding something funny or entertaining."],
            2: ["This text expresses anger, rage, fury, or strong displeasure and hostility."],
            3: ["This text expresses annoyance, irritation, frustration, or mild anger at something bothersome."],
            4: ["This text expresses approval, agreement, acceptance, or positive endorsement of something."],
            5: ["This text expresses caring, compassion, concern, empathy, or desire to help others."],
            6: ["This text expresses confusion, bewilderment, uncertainty, or lack of understanding."],
            7: ["This text expresses curiosity, interest, inquisitiveness, or desire to learn or know more."],
            8: ["This text expresses desire, wanting, longing, craving, or wish for something."],
            9: ["This text expresses disappointment, letdown, dissatisfaction, or unmet expectations."],
            10: ["This text expresses disapproval, disagreement, rejection, or negative judgment."],
            11: ["This text expresses disgust, revulsion, repulsion, or strong distaste."],
            12: ["This text expresses embarrassment, shame, awkwardness, or self-consciousness."],
            13: ["This text expresses excitement, enthusiasm, eagerness, or high energy and anticipation."],
            14: ["This text expresses fear, anxiety, worry, dread, or apprehension about something."],
            15: ["This text expresses gratitude, thankfulness, appreciation, or acknowledgment of kindness."],
            16: ["This text expresses grief, deep sorrow, mourning, or profound sadness over loss."],
            17: ["This text expresses joy, happiness, delight, pleasure, or positive emotional state."],
            18: ["This text expresses love, affection, fondness, deep caring, or romantic feelings."],
            19: ["This text expresses nervousness, anxiety, unease, jitters, or worried anticipation."],
            20: ["This text expresses optimism, hopefulness, positive outlook, or expectation of good outcomes."],
            21: ["This text expresses pride, satisfaction, accomplishment, or positive self-regard."],
            22: ["This text expresses realization, sudden understanding, epiphany, or coming to awareness."],
            23: ["This text expresses relief, ease, comfort, or release from worry or tension."],
            24: ["This text expresses remorse, regret, guilt, or sorrow for wrongdoing."],
            25: ["This text expresses sadness, sorrow, unhappiness, melancholy, or emotional pain."],
            26: ["This text expresses surprise, astonishment, shock, or unexpected reaction."],
            27: ["This text expresses neutral emotion, lacks strong emotional content, or is emotionally balanced."],
        },
    },
}


def get_label_texts(dataset_name: str, label_mode: str) -> Dict[int, List[str]]:
    """Get label texts for a dataset and mode.
    
    Args:
        dataset_name: Name of the dataset
        label_mode: Label mode (name_only, description, multi_description)
        
    Returns:
        Dictionary mapping label IDs to list of text representations
    """
    if dataset_name not in LABEL_SETS:
        raise ValueError(f"Dataset {dataset_name} not found in LABEL_SETS")
    
    if label_mode not in LABEL_SETS[dataset_name]:
        raise ValueError(f"Label mode {label_mode} not found for dataset {dataset_name}")
    
    return LABEL_SETS[dataset_name][label_mode]


def flatten_label_texts(label_dict: Dict[int, List[str]]) -> tuple:
    """Flatten label dictionary to lists.
    
    Args:
        label_dict: Dictionary mapping label IDs to text lists
        
    Returns:
        Tuple of (texts, label_ids) where each text maps to a label ID
    """
    texts = []
    label_ids = []
    
    for label_id, text_list in label_dict.items():
        for text in text_list:
            texts.append(text)
            label_ids.append(label_id)
    
    return texts, label_ids