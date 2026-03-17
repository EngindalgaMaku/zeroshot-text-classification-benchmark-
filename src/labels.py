"""Label definitions for zero-shot classification."""

from typing import Dict, List
import json
from pathlib import Path


# Cache for LLM-generated descriptions
_GENERATED_DESC_CACHE = None


def load_generated_descriptions(force_reload=False):
    """Load LLM-generated descriptions from JSON file.
    
    Args:
        force_reload: If True, bypass cache and reload from disk
    
    Returns:
        Dictionary with structure: {dataset: {label_id: {"l2": str, "l3": [str, str, str]}}}
    """
    global _GENERATED_DESC_CACHE
    if _GENERATED_DESC_CACHE is None or force_reload:
        desc_file = Path(__file__).parent / "label_descriptions" / "generated_descriptions.json"
        if desc_file.exists():
            with open(desc_file, encoding="utf-8") as f:
                _GENERATED_DESC_CACHE = json.load(f)
        else:
            _GENERATED_DESC_CACHE = {}
    return _GENERATED_DESC_CACHE


LABEL_SETS = {
    "ag_news": {
        "name_only": {
            0: ["World"],
            1: ["Sports"],
            2: ["Business"],
            3: ["Sci/Tech"],
        },
        "description": {
            0: ["This text is about international events, global politics, diplomacy, conflicts, or world affairs."],
            1: ["This text is about sports, matches, teams, athletes, tournaments, or competitions."],
            2: ["This text is about business, markets, finance, companies, trade, or the economy."],
            3: ["This text is about science, technology, computers, innovation, research, or digital products."],
        },
        "multi_description": {
            0: [
                "News about international events, global politics, or world affairs.",
                "Reports on international relations, diplomacy, or geopolitical developments.",
                "Coverage of global news, conflicts, or political events around the world.",
            ],
            1: [
                "News related to sports teams, athletes, or sporting events.",
                "Coverage of matches, competitions, or sports tournaments.",
                "Articles about sports performance, teams, or athletic competitions.",
            ],
            2: [
                "News about companies, markets, finance, or the global economy.",
                "Reports on business activity, corporations, or financial markets.",
                "Coverage of economic developments, companies, or business strategies.",
            ],
            3: [
                "News related to science, technology, computers, or innovation.",
                "Articles about scientific discoveries, research, or technological advances.",
                "Coverage of technology companies, software, or new scientific developments.",
            ],
        },
        "description_set_a": {
            0: ["Coverage of international affairs, geopolitical developments, and cross-border events shaping global society."],
            1: ["Reports on athletic competitions, team performance, player achievements, and organized sporting events."],
            2: ["Analysis of corporate activity, financial markets, economic trends, and commercial enterprise developments."],
            3: ["Coverage of scientific research, technological innovation, digital products, and engineering breakthroughs."],
        },
        "description_set_b": {
            0: ["News concerning international relations, foreign policy, armed conflicts, and events affecting multiple nations."],
            1: ["News concerning organized athletic competitions, sporting leagues, player transfers, and tournament outcomes."],
            2: ["News concerning corporate earnings, stock markets, mergers, trade policy, and macroeconomic indicators."],
            3: ["News concerning scientific discoveries, emerging technologies, space exploration, and digital innovation."],
        },
    },
    
    "dbpedia_14": {
        "name_only": {
            0: ["Company"],
            1: ["EducationalInstitution"],
            2: ["Artist"],
            3: ["Athlete"],
            4: ["OfficeHolder"],
            5: ["MeanOfTransportation"],
            6: ["Building"],
            7: ["NaturalPlace"],
            8: ["Village"],
            9: ["Animal"],
            10: ["Plant"],
            11: ["Album"],
            12: ["Film"],
            13: ["WrittenWork"],
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
        "multi_description": {
            0: [
                "An article about a company, business organization, or corporate entity.",
                "Text describing a corporation, firm, or commercial enterprise.",
                "Content about a business, startup, or industrial organization.",
            ],
            1: [
                "An article about a school, university, college, or educational institution.",
                "Text describing an academy, institute, or place of learning.",
                "Content about an educational organization, faculty, or academic body.",
            ],
            2: [
                "An article about an artist, musician, painter, or creative professional.",
                "Text describing a performer, sculptor, or person known for artistic work.",
                "Content about a creative individual in music, visual arts, or performing arts.",
            ],
            3: [
                "An article about an athlete, sports player, or competitive sportsperson.",
                "Text describing a professional or amateur competitor in sports.",
                "Content about a person known for athletic performance or sports career.",
            ],
            4: [
                "An article about a politician, government official, or elected representative.",
                "Text describing a public office holder, minister, or civil servant.",
                "Content about a person holding a political or governmental position.",
            ],
            5: [
                "An article about a vehicle, transportation system, or mode of travel.",
                "Text describing a ship, aircraft, train, or other means of transportation.",
                "Content about a transit system, vehicle type, or transport infrastructure.",
            ],
            6: [
                "An article about a building, structure, or architectural landmark.",
                "Text describing a constructed facility, monument, or architectural work.",
                "Content about a physical structure such as a bridge, tower, or stadium.",
            ],
            7: [
                "An article about a natural place, landscape, or geographical feature.",
                "Text describing a mountain, river, lake, or natural formation.",
                "Content about a natural environment, wilderness area, or geographic location.",
            ],
            8: [
                "An article about a village, small town, or rural settlement.",
                "Text describing a community, hamlet, or small populated locality.",
                "Content about a minor administrative division or small geographic community.",
            ],
            9: [
                "An article about an animal species, creature, or wildlife.",
                "Text describing a mammal, bird, reptile, or other living organism.",
                "Content about an animal's biology, habitat, or classification.",
            ],
            10: [
                "An article about a plant species, tree, flower, or botanical entity.",
                "Text describing vegetation, flora, or a specific plant organism.",
                "Content about a plant's characteristics, habitat, or taxonomy.",
            ],
            11: [
                "An article about a music album, record, or collection of songs.",
                "Text describing a musical release, discography entry, or studio album.",
                "Content about an album's tracks, artists, or production.",
            ],
            12: [
                "An article about a film, movie, or cinema production.",
                "Text describing a motion picture, documentary, or cinematic work.",
                "Content about a film's plot, cast, director, or release.",
            ],
            13: [
                "An article about a book, novel, or written literary work.",
                "Text describing a publication, manuscript, or written document.",
                "Content about a written work's author, genre, or subject matter.",
            ],
        },
        "description_set_a": {
            0: ["A commercial entity organized to produce goods or services for profit within an economic system."],
            1: ["An organization established to provide formal instruction, training, or academic learning to students."],
            2: ["An individual recognized for creative expression through visual, performing, or literary artistic disciplines."],
            3: ["A person who engages in competitive physical activities, sports, or organized athletic pursuits professionally."],
            4: ["An individual who holds an appointed or elected position within a governmental or institutional body."],
            5: ["A vehicle, vessel, or system designed to convey people or goods from one location to another."],
            6: ["A permanent structure constructed for human use, habitation, commerce, or institutional purposes."],
            7: ["A geographic feature or landscape formed by natural processes, unaltered by significant human intervention."],
            8: ["A small, sparsely populated human settlement, typically rural and subordinate to larger administrative units."],
            9: ["A multicellular organism belonging to the kingdom Animalia, capable of voluntary movement and sensation."],
            10: ["A photosynthetic organism belonging to the plant kingdom, typically rooted and producing its own nutrients."],
            11: ["A collection of audio recordings released as a unified work by a musical artist or group."],
            12: ["A narrative or documentary work produced for cinematic or screen-based visual presentation to audiences."],
            13: ["A text-based composition intended for reading, encompassing fiction, nonfiction, poetry, or scholarly works."],
        },
        "description_set_b": {
            0: ["An organization or association engaged in commercial, industrial, or professional activities as a legal entity."],
            1: ["An institution providing systematic instruction in academic disciplines, typically granting recognized qualifications."],
            2: ["A person who creates works of aesthetic value in fields such as music, visual arts, or performance."],
            3: ["A person who participates in sports or physical exercises, especially in competitive professional contexts."],
            4: ["A person who occupies a position of authority or trust in a public, governmental, or organizational role."],
            5: ["A conveyance or infrastructure system used to transport passengers or freight across distances."],
            6: ["A man-made structure with a roof and walls, serving residential, commercial, industrial, or civic functions."],
            7: ["A physical feature of the earth's surface shaped by geological or biological processes rather than human activity."],
            8: ["A small clustered human settlement, smaller than a town, typically located in a rural or semi-rural area."],
            9: ["A living organism of the kingdom Animalia, distinguished by eukaryotic cells and heterotrophic metabolism."],
            10: ["A living organism of the kingdom Plantae, typically producing energy through photosynthesis and lacking mobility."],
            11: ["A published recording comprising a set of musical tracks issued together as a single commercial release."],
            12: ["A sequence of moving images recorded and edited to tell a story or document events for theatrical release."],
            13: ["A document or literary composition produced in written or printed form, intended for reading by an audience."],
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
        "multi_description": {
            0: [
                "A question about society, culture, traditions, or social norms.",
                "Text discussing cultural practices, social issues, or community values.",
                "Content about human society, cultural identity, or social behavior.",
            ],
            1: [
                "A question about science, mathematics, or scientific concepts.",
                "Text discussing physics, chemistry, biology, or mathematical problems.",
                "Content about scientific research, experiments, or natural phenomena.",
            ],
            2: [
                "A question about health, medicine, or wellness.",
                "Text discussing diseases, treatments, nutrition, or medical conditions.",
                "Content about physical or mental health, healthcare, or medical advice.",
            ],
            3: [
                "A question about education, learning, or academic topics.",
                "Text discussing schools, teaching methods, or educational resources.",
                "Content about academic subjects, study tips, or educational institutions.",
            ],
            4: [
                "A question about computers, internet, or technology.",
                "Text discussing software, hardware, programming, or digital services.",
                "Content about tech products, online platforms, or computer systems.",
            ],
            5: [
                "A question about sports, athletics, or physical competitions.",
                "Text discussing games, teams, players, or sporting events.",
                "Content about sports rules, performance, or athletic training.",
            ],
            6: [
                "A question about business, finance, or economics.",
                "Text discussing investing, markets, companies, or financial planning.",
                "Content about commercial activities, economic trends, or business strategy.",
            ],
            7: [
                "A question about entertainment, music, movies, or leisure.",
                "Text discussing celebrities, TV shows, arts, or popular culture.",
                "Content about recreational activities, media, or cultural entertainment.",
            ],
            8: [
                "A question about family, relationships, or interpersonal matters.",
                "Text discussing marriage, parenting, friendships, or personal connections.",
                "Content about family dynamics, romantic relationships, or social bonds.",
            ],
            9: [
                "A question about politics, government, or public policy.",
                "Text discussing laws, elections, political parties, or governance.",
                "Content about political issues, civic matters, or governmental decisions.",
            ],
        },
        "description_set_a": {
            0: ["Questions addressing human social organization, cultural traditions, values, and collective behavioral norms."],
            1: ["Questions exploring empirical inquiry, mathematical reasoning, natural phenomena, and scientific methodology."],
            2: ["Questions concerning physical or mental well-being, medical conditions, treatments, and healthcare practices."],
            3: ["Questions about formal learning systems, academic subjects, instructional methods, and knowledge resources."],
            4: ["Questions relating to computing devices, software applications, internet services, and digital technologies."],
            5: ["Questions about organized physical competitions, athletic performance, teams, and sporting rules or events."],
            6: ["Questions addressing commercial enterprises, financial instruments, economic systems, and investment strategies."],
            7: ["Questions about popular media, musical artists, film, television, and recreational cultural activities."],
            8: ["Questions concerning domestic relationships, parenting, romantic partnerships, and interpersonal social bonds."],
            9: ["Questions about governmental structures, political processes, public policy, and civic institutions."],
        },
        "description_set_b": {
            0: ["Questions about the customs, institutions, values, and shared practices that define human communities."],
            1: ["Questions about the systematic study of the natural world through observation, experiment, and mathematical analysis."],
            2: ["Questions about the state of physical and mental well-being and the practices used to maintain or restore it."],
            3: ["Questions about the process of acquiring knowledge and skills through formal instruction or reference materials."],
            4: ["Questions about electronic computing systems, networked communication, and digital information technologies."],
            5: ["Questions about competitive physical activities governed by rules, involving individual or team participation."],
            6: ["Questions about the production, distribution, and exchange of goods and services within economic systems."],
            7: ["Questions about creative works, performances, and media produced for public enjoyment and cultural engagement."],
            8: ["Questions about the social bonds, obligations, and emotional connections among relatives and intimate partners."],
            9: ["Questions about the systems and processes by which societies are organized, governed, and administered."],
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
            53: ["reverted card payment?"],
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
        "multi_description": {
            0: ["User wants to activate their card.", "How to activate a new bank card.", "Asking about card activation process."],
            1: ["User asking about age restrictions for services.", "Question about minimum age requirements.", "Inquiry about age eligibility for account features."],
            2: ["User has a question about Apple Pay or Google Pay.", "Asking about mobile payment integration.", "Issue or question about digital wallet support."],
            3: ["User needs help with ATM access or support.", "Question about ATM locations or functionality.", "Asking about cash machine availability."],
            4: ["User wants to enable automatic top up.", "Question about setting up auto top-up.", "Asking how automatic balance refill works."],
            5: ["Balance not updated after a bank transfer.", "Transfer completed but balance unchanged.", "Asking why balance didn't reflect a bank transfer."],
            6: ["Balance not updated after depositing cheque or cash.", "Deposit made but balance not showing.", "Asking why cash or cheque deposit isn't reflected."],
            7: ["User cannot add a beneficiary.", "Beneficiary addition is blocked or not allowed.", "Asking why a recipient cannot be added."],
            8: ["User wants to cancel a pending transfer.", "Asking how to stop or reverse a transfer.", "Request to cancel a money transfer."],
            9: ["User's card is about to expire.", "Asking about card expiry and renewal.", "Card expiration date approaching, needs information."],
            10: ["User asking where their card is accepted.", "Question about card acceptance at merchants.", "Asking about card compatibility with payment terminals."],
            11: ["User asking when their card will arrive.", "Question about card delivery status.", "Waiting for card and asking about arrival time."],
            12: ["User wants an estimated delivery time for their card.", "Asking how long card delivery takes.", "Question about expected card arrival date."],
            13: ["User has trouble linking their card.", "Card linking issue or failure.", "Asking how to connect a card to the account."],
            14: ["User's card is not working.", "Card declined or malfunctioning.", "Asking why card isn't functioning properly."],
            15: ["User was charged a fee for a card payment.", "Unexpected fee on card transaction.", "Asking about charges applied to card payments."],
            16: ["Card payment not recognised or recorded.", "Transaction missing from account history.", "Asking why a card payment doesn't appear."],
            17: ["Wrong exchange rate applied to card payment.", "Card payment charged at incorrect rate.", "Asking about exchange rate discrepancy on payment."],
            18: ["Card was swallowed by an ATM.", "ATM retained the user's card.", "Asking what to do after ATM kept the card."],
            19: ["User was charged for a cash withdrawal.", "Unexpected fee on ATM withdrawal.", "Asking about cash withdrawal charges."],
            20: ["Cash withdrawal not recognised in account.", "ATM withdrawal not showing in history.", "Asking why a cash withdrawal isn't recorded."],
            21: ["User wants to change their PIN.", "Asking how to update or reset PIN code.", "Request to modify card PIN."],
            22: ["User thinks their card is compromised.", "Suspected card fraud or security breach.", "Asking about compromised or stolen card."],
            23: ["Contactless payment not working.", "Tap-to-pay feature failing.", "Asking why contactless transactions are declined."],
            24: ["User asking about country support.", "Question about international card usage.", "Asking which countries the card works in."],
            25: ["Card payment was declined.", "Transaction rejected at point of sale.", "Asking why a card payment was refused."],
            26: ["Cash withdrawal was declined.", "ATM refused to dispense cash.", "Asking why cash withdrawal was rejected."],
            27: ["Transfer was declined.", "Money transfer rejected or blocked.", "Asking why a transfer didn't go through."],
            28: ["Direct debit payment not recognised.", "Direct debit missing from account.", "Asking why a direct debit isn't showing."],
            29: ["User asking about disposable card limits.", "Question about virtual card spending limits.", "Asking about restrictions on disposable cards."],
            30: ["User wants to edit personal details.", "Asking how to update account information.", "Request to change name, address, or contact info."],
            31: ["User asking about exchange charges.", "Question about fees on currency exchange.", "Asking about costs for converting currency."],
            32: ["User asking about exchange rates.", "Question about currency conversion rates.", "Asking what rate applies to foreign transactions."],
            33: ["User wants to exchange currency via the app.", "Asking how to convert money in the app.", "Request to use in-app currency exchange."],
            34: ["User sees an unexpected charge on statement.", "Extra or unknown fee on account.", "Asking about an unrecognised charge."],
            35: ["User's transfer has failed.", "Money transfer did not complete.", "Asking why a transfer failed."],
            36: ["User asking about fiat currency support.", "Question about supported traditional currencies.", "Asking which fiat currencies are available."],
            37: ["User wants a disposable virtual card.", "Asking how to get a one-time use card.", "Request for a temporary virtual card."],
            38: ["User wants to get a physical card.", "Asking how to order a plastic card.", "Request for a physical bank card."],
            39: ["User wants a spare card.", "Asking about getting an additional card.", "Request for a backup payment card."],
            40: ["User wants a virtual card.", "Asking how to get a digital card.", "Request for a virtual payment card."],
            41: ["User lost their card or it was stolen.", "Card missing or taken by someone.", "Asking what to do after losing a card."],
            42: ["User lost their phone or it was stolen.", "Phone missing or taken.", "Asking what to do after losing a phone."],
            43: ["User wants to order a physical card.", "Asking how to request a new card.", "Request to have a card sent by mail."],
            44: ["User forgot their passcode.", "Asking how to reset or recover passcode.", "Locked out due to forgotten PIN or password."],
            45: ["Card payment is pending.", "Transaction showing as pending.", "Asking why a card payment hasn't cleared."],
            46: ["Cash withdrawal is pending.", "ATM transaction still pending.", "Asking why a withdrawal hasn't been processed."],
            47: ["Top up is pending.", "Balance top-up not yet reflected.", "Asking why a top-up is still processing."],
            48: ["Transfer is pending.", "Money transfer not yet completed.", "Asking why a transfer is still pending."],
            49: ["User's PIN is blocked.", "PIN locked after too many attempts.", "Asking how to unblock a blocked PIN."],
            50: ["User asking about receiving money.", "Question about incoming transfers.", "Asking how to receive payments or transfers."],
            51: ["Refund not showing in account.", "Expected refund hasn't appeared.", "Asking why a refund isn't visible."],
            52: ["User wants to request a refund.", "Asking how to get money back.", "Request to initiate a refund process."],
            53: ["Card payment was reverted.", "Transaction reversed or undone.", "Asking about a reverted card payment."],
            54: ["User asking about supported cards and currencies.", "Question about accepted card types.", "Asking which cards and currencies are supported."],
            55: ["User wants to close their account.", "Asking how to terminate the account.", "Request to delete or cancel the account."],
            56: ["User charged for top up via bank transfer.", "Fee applied to bank transfer top-up.", "Asking about charges on bank transfer top-ups."],
            57: ["User charged for top up by card.", "Fee on card top-up transaction.", "Asking about costs for topping up by card."],
            58: ["User wants to top up with cash or cheque.", "Asking how to add funds via cash or cheque.", "Request for cash or cheque deposit top-up."],
            59: ["User's top up failed.", "Top-up attempt did not succeed.", "Asking why a top-up didn't go through."],
            60: ["User asking about top up limits.", "Question about maximum top-up amounts.", "Asking about restrictions on adding funds."],
            61: ["User's top up was reverted.", "Top-up reversed or cancelled.", "Asking why a top-up was undone."],
            62: ["User asking about topping up by card.", "Question about card top-up process.", "Asking how to add funds using a card."],
            63: ["User was charged twice for a transaction.", "Duplicate charge on account.", "Asking about a double payment."],
            64: ["User was charged a transfer fee.", "Fee applied to a money transfer.", "Asking about costs for sending money."],
            65: ["User asking about receiving a transfer.", "Question about incoming money transfer.", "Asking how transfers are received into account."],
            66: ["Transfer not received by the recipient.", "Sent money hasn't arrived.", "Asking why recipient didn't get the transfer."],
            67: ["User asking about transfer timing.", "Question about how long transfers take.", "Asking about expected transfer processing time."],
            68: ["User unable to verify their identity.", "Identity verification failing.", "Asking why identity check isn't working."],
            69: ["User wants to verify their identity.", "Asking how to complete identity verification.", "Request to submit identity documents."],
            70: ["User needs to verify source of funds.", "Asking about source of funds verification.", "Request to confirm where funds come from."],
            71: ["User needs to verify a top-up.", "Asking about top-up verification process.", "Request to confirm a top-up transaction."],
            72: ["Virtual card not working.", "Digital card failing at checkout.", "Asking why virtual card is being declined."],
            73: ["User asking about Visa or Mastercard.", "Question about card network type.", "Asking whether card is Visa or Mastercard."],
            74: ["User asking why identity verification is needed.", "Question about purpose of ID check.", "Asking why they need to verify identity."],
            75: ["User received wrong amount of cash.", "ATM dispensed incorrect amount.", "Asking about cash withdrawal amount discrepancy."],
            76: ["Wrong exchange rate for cash withdrawal.", "Incorrect rate applied to ATM withdrawal.", "Asking about exchange rate on cash withdrawal."],
        },
        "description_set_a": {
            0: ["A request to enable a newly issued payment card for use in transactions and purchases."],
            1: ["An inquiry about minimum age requirements for accessing specific financial products or services."],
            2: ["A question about integrating or using a mobile digital wallet payment service on a device."],
            3: ["A request for information about automated teller machine availability, compatibility, or functionality."],
            4: ["A query about configuring a recurring automatic balance replenishment feature on an account."],
            5: ["A report that an account balance remains unchanged following a completed interbank transfer."],
            6: ["A report that an account balance was not updated after depositing a physical cheque or cash."],
            7: ["A complaint that a designated payee cannot be added or is restricted from receiving transfers."],
            8: ["A request to stop or reverse a money transfer that has not yet been fully processed."],
            9: ["A notification or inquiry about a payment card approaching its printed expiration date."],
            10: ["An inquiry about which merchants, terminals, or payment networks accept a particular card."],
            11: ["A question about the current delivery status or whereabouts of a newly ordered payment card."],
            12: ["A request for an estimated timeframe for a newly ordered payment card to be delivered."],
            13: ["A problem or question about connecting a payment card to an account or digital service."],
            14: ["A report that a payment card is failing to process transactions or function as expected."],
            15: ["A complaint about an unexpected service charge applied to a card-based payment transaction."],
            16: ["A report that a completed card payment does not appear in the account transaction history."],
            17: ["A complaint that an incorrect currency conversion rate was applied to a card payment abroad."],
            18: ["A report that an automated teller machine retained and did not return a payment card."],
            19: ["A complaint about a fee charged for withdrawing physical cash from an ATM or branch."],
            20: ["A report that a completed cash withdrawal does not appear in the account transaction history."],
            21: ["A request to update or reset the personal identification number associated with a payment card."],
            22: ["A concern that a payment card's security has been breached or used without authorization."],
            23: ["A report that tap-to-pay or near-field communication payment functionality is not operating."],
            24: ["An inquiry about whether a card or account can be used in a specific foreign country."],
            25: ["A report that a card payment was refused at a merchant terminal or online checkout."],
            26: ["A report that an attempt to withdraw cash from an ATM was refused or blocked."],
            27: ["A report that a money transfer was rejected and did not reach the intended recipient."],
            28: ["A report that a scheduled direct debit payment is absent from the account transaction record."],
            29: ["An inquiry about spending or usage restrictions applied to a temporary single-use virtual card."],
            30: ["A request to update personal information such as name, address, or contact details on an account."],
            31: ["An inquiry about fees charged when converting one currency to another through the service."],
            32: ["An inquiry about the current rate applied when converting between two different currencies."],
            33: ["A request to perform a currency conversion transaction using the mobile application interface."],
            34: ["A complaint about an unrecognized or unexpected charge appearing on an account statement."],
            35: ["A report that a money transfer did not complete and the funds were not delivered."],
            36: ["An inquiry about which traditional government-issued currencies are supported by the service."],
            37: ["A request to obtain a temporary, single-use virtual card for secure online transactions."],
            38: ["A request to receive a tangible plastic payment card delivered by postal mail."],
            39: ["A request to obtain an additional backup payment card linked to an existing account."],
            40: ["A request to create or obtain a digital card for use in online or contactless payments."],
            41: ["A report that a physical payment card has been misplaced or taken without authorization."],
            42: ["A report that a mobile phone has been misplaced or taken, affecting account access."],
            43: ["A request to place an order for a new physical payment card to be mailed to the user."],
            44: ["A report of being unable to recall the security code needed to access an account."],
            45: ["A report that a card payment has been authorized but not yet fully settled or cleared."],
            46: ["A report that a cash withdrawal transaction is authorized but not yet fully processed."],
            47: ["A report that a balance top-up has been initiated but not yet reflected in the account."],
            48: ["A report that a money transfer has been initiated but not yet completed or delivered."],
            49: ["A report that a personal identification number has been locked after repeated failed attempts."],
            50: ["An inquiry about how to receive incoming money transfers or payments into an account."],
            51: ["A report that an expected refund has not yet appeared in the account balance."],
            52: ["A request to initiate the process of returning funds for a previous transaction."],
            53: ["A report that a previously completed card payment has been reversed or cancelled."],
            54: ["An inquiry about which card types and currency denominations are accepted by the service."],
            55: ["A request to permanently close and deactivate an existing financial account."],
            56: ["A complaint about a fee charged when adding funds via an interbank transfer method."],
            57: ["A complaint about a fee charged when adding funds using a debit or credit card."],
            58: ["A request for information about adding funds to an account using physical cash or a cheque."],
            59: ["A report that an attempt to add funds to an account balance did not succeed."],
            60: ["An inquiry about the maximum or minimum amounts permitted when adding funds to an account."],
            61: ["A report that a previously completed top-up transaction has been reversed or cancelled."],
            62: ["An inquiry about the process or steps involved in adding funds using a payment card."],
            63: ["A report that the same transaction has been charged to an account on two separate occasions."],
            64: ["A complaint about a service fee applied to a money transfer sent from an account."],
            65: ["An inquiry about how incoming money transfers are credited to a receiving account."],
            66: ["A report that a sent transfer has not been received by the intended destination account."],
            67: ["An inquiry about how long a money transfer takes to reach the recipient's account."],
            68: ["A report of being unable to complete a required identity verification process successfully."],
            69: ["A request to complete or submit documentation for an account identity verification process."],
            70: ["A request to provide documentation confirming the origin of funds held in an account."],
            71: ["A request to confirm or authenticate a specific top-up transaction for compliance purposes."],
            72: ["A report that a digital card is failing to process payments or function as expected."],
            73: ["An inquiry about whether a card operates on the Visa or Mastercard payment network."],
            74: ["An inquiry about the reasons or regulatory requirements behind mandatory identity verification."],
            75: ["A report that an ATM dispensed a different amount of cash than was requested."],
            76: ["A complaint that an incorrect currency conversion rate was applied to a cash withdrawal."],
        },
        "description_set_b": {
            0: ["Customer intent to initiate the activation procedure for a newly received payment card."],
            1: ["Customer inquiry regarding minimum age eligibility criteria for account opening or feature access."],
            2: ["Customer inquiry or issue relating to the setup or use of Apple Pay or Google Pay services."],
            3: ["Customer inquiry about the availability, compatibility, or functionality of ATM cash machines."],
            4: ["Customer inquiry about enabling a feature that automatically replenishes account balance when low."],
            5: ["Customer report that account balance has not been updated following a completed bank transfer."],
            6: ["Customer report that account balance has not been updated after a cheque or cash deposit."],
            7: ["Customer report that a designated payee or beneficiary cannot be added to the account."],
            8: ["Customer request to stop or reverse a transfer that has not yet been fully processed."],
            9: ["Customer inquiry about a payment card that is approaching or has reached its expiry date."],
            10: ["Customer inquiry about the merchants, networks, or terminals where their card is accepted."],
            11: ["Customer inquiry about the current delivery status of a newly ordered payment card."],
            12: ["Customer inquiry about the expected timeframe for a newly ordered card to be delivered."],
            13: ["Customer issue or inquiry about connecting a payment card to their account or service."],
            14: ["Customer report that their payment card is not functioning correctly at point of sale."],
            15: ["Customer complaint about an unexpected fee charged on a card-based payment transaction."],
            16: ["Customer report that a completed card payment does not appear in their transaction history."],
            17: ["Customer complaint that an incorrect exchange rate was applied to a card payment abroad."],
            18: ["Customer report that an ATM has retained their payment card without returning it."],
            19: ["Customer complaint about a fee charged for withdrawing cash from an ATM or branch."],
            20: ["Customer report that a completed cash withdrawal does not appear in their account history."],
            21: ["Customer request to update or reset the PIN associated with their payment card."],
            22: ["Customer concern that their card details have been stolen or used without authorization."],
            23: ["Customer report that the contactless or tap-to-pay feature on their card is not working."],
            24: ["Customer inquiry about whether their card or account can be used in a specific country."],
            25: ["Customer report that a card payment was declined at a merchant or online checkout."],
            26: ["Customer report that an ATM declined to dispense cash when requested."],
            27: ["Customer report that a money transfer was rejected and did not reach the recipient."],
            28: ["Customer report that a direct debit payment is missing from their account transaction record."],
            29: ["Customer inquiry about the spending limits or restrictions on a disposable virtual card."],
            30: ["Customer request to update personal information such as name, address, or phone number."],
            31: ["Customer inquiry about fees charged when performing a currency exchange transaction."],
            32: ["Customer inquiry about the current rate applied when converting between currencies."],
            33: ["Customer request to perform a currency exchange using the in-app exchange feature."],
            34: ["Customer complaint about an unrecognized or unexpected charge on their account statement."],
            35: ["Customer report that a money transfer failed and the funds were not delivered."],
            36: ["Customer inquiry about which traditional fiat currencies are supported by the service."],
            37: ["Customer request to obtain a temporary, single-use virtual card for online transactions."],
            38: ["Customer request to receive a physical plastic payment card by postal delivery."],
            39: ["Customer request to obtain an additional backup card linked to their existing account."],
            40: ["Customer request to create or obtain a virtual card for digital or contactless payments."],
            41: ["Customer report that their physical payment card has been lost or stolen."],
            42: ["Customer report that their mobile phone has been lost or stolen, affecting account access."],
            43: ["Customer request to place an order for a new physical card to be mailed to them."],
            44: ["Customer report of being unable to recall the passcode needed to access their account."],
            45: ["Customer report that a card payment has been authorized but not yet fully settled."],
            46: ["Customer report that a cash withdrawal transaction is authorized but not yet processed."],
            47: ["Customer report that a balance top-up has been initiated but not yet reflected."],
            48: ["Customer report that a money transfer has been initiated but not yet completed."],
            49: ["Customer report that their PIN has been locked after too many incorrect attempts."],
            50: ["Customer inquiry about how to receive incoming money transfers or payments."],
            51: ["Customer report that an expected refund has not yet appeared in their account."],
            52: ["Customer request to initiate a refund for a previous transaction."],
            53: ["Customer report that a previously completed card payment has been reversed."],
            54: ["Customer inquiry about which card types and currencies are accepted by the service."],
            55: ["Customer request to permanently close and deactivate their financial account."],
            56: ["Customer complaint about a fee charged when topping up via bank transfer."],
            57: ["Customer complaint about a fee charged when topping up using a debit or credit card."],
            58: ["Customer inquiry about adding funds to their account using physical cash or a cheque."],
            59: ["Customer report that an attempt to add funds to their account balance did not succeed."],
            60: ["Customer inquiry about the maximum or minimum amounts permitted for account top-ups."],
            61: ["Customer report that a previously completed top-up transaction has been reversed."],
            62: ["Customer inquiry about the process or steps for adding funds using a payment card."],
            63: ["Customer report that the same transaction has been charged to their account twice."],
            64: ["Customer complaint about a service fee applied to a money transfer."],
            65: ["Customer inquiry about how incoming money transfers are credited to their account."],
            66: ["Customer report that a sent transfer has not been received by the intended recipient."],
            67: ["Customer inquiry about how long a money transfer takes to reach the recipient."],
            68: ["Customer report of being unable to complete a required identity verification process."],
            69: ["Customer request to complete or submit documentation for identity verification."],
            70: ["Customer request to provide documentation confirming the origin of their funds."],
            71: ["Customer request to confirm or authenticate a specific top-up transaction."],
            72: ["Customer report that their virtual card is failing to process payments correctly."],
            73: ["Customer inquiry about whether their card operates on the Visa or Mastercard network."],
            74: ["Customer inquiry about the reasons or requirements behind mandatory identity verification."],
            75: ["Customer report that an ATM dispensed a different amount of cash than was requested."],
            76: ["Customer complaint that an incorrect exchange rate was applied to a cash withdrawal."],
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
        "multi_description": {
            0: [
                "This text expresses a negative or pessimistic view of financial markets.",
                "Content suggesting declining prices, losses, or bearish market outlook.",
                "Text indicating negative sentiment, downward trends, or financial pessimism.",
            ],
            1: [
                "This text expresses a positive or optimistic view of financial markets.",
                "Content suggesting rising prices, gains, or bullish market outlook.",
                "Text indicating positive sentiment, upward trends, or financial optimism.",
            ],
            2: [
                "This text expresses a neutral or balanced view of financial markets.",
                "Content reporting facts without clear positive or negative bias.",
                "Text with objective, impartial, or mixed financial sentiment.",
            ],
        },
        "description_set_a": {
            0: ["Text conveying a pessimistic financial outlook, anticipating declining asset prices or deteriorating market conditions."],
            1: ["Text conveying an optimistic financial outlook, anticipating rising asset prices or improving market conditions."],
            2: ["Text presenting financial information without expressing a directional market opinion or emotional bias."],
        },
        "description_set_b": {
            0: ["Text reflecting a negative psychological disposition toward financial assets, expecting prices to fall."],
            1: ["Text reflecting a positive psychological disposition toward financial assets, expecting prices to rise."],
            2: ["Text reflecting an absence of directional emotional bias, presenting financial information objectively."],
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
        "multi_description": {
            0: [
                "Text discussing atheism, non-belief, or secular philosophy.",
                "Content about religious skepticism or rejection of theism.",
                "Discussion of atheist views, humanism, or non-religious worldviews.",
            ],
            1: [
                "Text about computer graphics, rendering, or image processing.",
                "Content discussing visualization software or graphical algorithms.",
                "Discussion of 3D graphics, image formats, or display technology.",
            ],
            2: [
                "Text about Microsoft Windows operating system.",
                "Content discussing Windows OS issues, tips, or configuration.",
                "Discussion of Windows software, drivers, or system problems.",
            ],
            3: [
                "Text about IBM PC hardware components or upgrades.",
                "Content discussing PC parts, peripherals, or technical specs.",
                "Discussion of IBM-compatible hardware, CPUs, or expansion cards.",
            ],
            4: [
                "Text about Apple Macintosh hardware or components.",
                "Content discussing Mac hardware, specs, or upgrades.",
                "Discussion of Apple computer hardware, peripherals, or compatibility.",
            ],
            5: [
                "Text about the X Window System or Unix graphical interface.",
                "Content discussing X11, window managers, or Unix GUI software.",
                "Discussion of X display server, desktop environments, or Unix graphics.",
            ],
            6: [
                "Text that is a for-sale listing or marketplace advertisement.",
                "Content offering items for purchase or trade.",
                "Discussion of selling, buying, or trading goods.",
            ],
            7: [
                "Text about automobiles, cars, or automotive topics.",
                "Content discussing vehicles, driving, or car maintenance.",
                "Discussion of car models, repairs, or automotive technology.",
            ],
            8: [
                "Text about motorcycles, bikes, or riding.",
                "Content discussing motorcycle models, maintenance, or riding tips.",
                "Discussion of motorbikes, scooters, or motorcycle culture.",
            ],
            9: [
                "Text about baseball, MLB, or baseball players.",
                "Content discussing baseball games, teams, or statistics.",
                "Discussion of baseball rules, seasons, or player performance.",
            ],
            10: [
                "Text about hockey, NHL, or ice hockey.",
                "Content discussing hockey games, teams, or players.",
                "Discussion of hockey rules, seasons, or player statistics.",
            ],
            11: [
                "Text about cryptography, encryption, or security algorithms.",
                "Content discussing ciphers, keys, or cryptographic protocols.",
                "Discussion of data security, encryption methods, or cryptanalysis.",
            ],
            12: [
                "Text about electronics, circuits, or electrical engineering.",
                "Content discussing electronic components, devices, or schematics.",
                "Discussion of circuit design, semiconductors, or electrical systems.",
            ],
            13: [
                "Text about medicine, health conditions, or medical research.",
                "Content discussing diseases, treatments, or healthcare.",
                "Discussion of medical science, drugs, or clinical topics.",
            ],
            14: [
                "Text about space, astronomy, or space exploration.",
                "Content discussing planets, stars, NASA, or astrophysics.",
                "Discussion of space missions, telescopes, or cosmology.",
            ],
            15: [
                "Text about Christianity, Christian faith, or the Bible.",
                "Content discussing Christian theology, prayer, or church.",
                "Discussion of Christian beliefs, scripture, or religious practice.",
            ],
            16: [
                "Text about gun rights, firearms, or gun control.",
                "Content discussing weapons policy, Second Amendment, or gun ownership.",
                "Discussion of firearms regulation, gun culture, or shooting sports.",
            ],
            17: [
                "Text about Middle East politics or regional conflicts.",
                "Content discussing Israel, Palestine, or Middle Eastern affairs.",
                "Discussion of geopolitical tensions, wars, or diplomacy in the Middle East.",
            ],
            18: [
                "Text about general political topics or political debates.",
                "Content discussing political parties, policies, or civic issues.",
                "Discussion of government, elections, or miscellaneous political matters.",
            ],
            19: [
                "Text about general religious topics or interfaith discussion.",
                "Content discussing spirituality, faith, or religious philosophy.",
                "Discussion of religion, theology, or miscellaneous spiritual matters.",
            ],
        },
        "description_set_a": {
            0: ["Discussion of non-theistic worldviews, rejection of religious belief, and secular philosophical perspectives."],
            1: ["Technical discussion of digital image creation, rendering algorithms, and computer-generated visual content."],
            2: ["Discussion of issues, configurations, and troubleshooting related to the Microsoft Windows operating system."],
            3: ["Discussion of hardware components, peripherals, and technical specifications for IBM-compatible personal computers."],
            4: ["Discussion of hardware components, peripherals, and technical specifications for Apple Macintosh computers."],
            5: ["Discussion of the X Window System graphical interface, display servers, and Unix desktop environments."],
            6: ["Posts offering items for sale, trade, or purchase within a community marketplace context."],
            7: ["Discussion of motor vehicles, automotive technology, driving experiences, and car maintenance practices."],
            8: ["Discussion of motorcycles, riding techniques, bike maintenance, and motorcycle culture or community."],
            9: ["Discussion of professional baseball leagues, teams, player statistics, and game results or analysis."],
            10: ["Discussion of professional ice hockey leagues, teams, player statistics, and game results or analysis."],
            11: ["Discussion of cryptographic algorithms, encryption protocols, data security, and privacy technologies."],
            12: ["Discussion of electronic circuits, components, devices, and principles of electrical engineering."],
            13: ["Discussion of medical conditions, treatments, pharmaceutical research, and healthcare practices or policies."],
            14: ["Discussion of space exploration, astronomical phenomena, planetary science, and astrophysical research."],
            15: ["Discussion of Christian theology, biblical interpretation, religious practice, and faith-based community life."],
            16: ["Discussion of firearms policy, gun ownership rights, weapons regulation, and Second Amendment debates."],
            17: ["Discussion of political conflicts, diplomatic relations, and geopolitical developments in the Middle East region."],
            18: ["Discussion of general political topics, policy debates, electoral issues, and civic governance matters."],
            19: ["Discussion of diverse religious traditions, spiritual beliefs, theological questions, and interfaith perspectives."],
        },
        "description_set_b": {
            0: ["Posts discussing the philosophical position that deities do not exist and critiquing religious belief systems."],
            1: ["Posts discussing techniques, software, and algorithms for creating and manipulating digital visual imagery."],
            2: ["Posts discussing problems, configurations, and usage of the Microsoft Windows personal computer platform."],
            3: ["Posts discussing hardware components, compatibility, and upgrades for IBM-compatible personal computers."],
            4: ["Posts discussing hardware components, peripherals, and technical issues specific to Apple Macintosh systems."],
            5: ["Posts discussing the X Window System, a network-transparent graphical interface for Unix-like operating systems."],
            6: ["Posts advertising items available for purchase, trade, or auction within an online community forum."],
            7: ["Posts discussing passenger vehicles, automotive engineering, road travel, and car ownership experiences."],
            8: ["Posts discussing two-wheeled motorized vehicles, riding practices, and motorcycle enthusiast culture."],
            9: ["Posts discussing the sport of baseball, including Major League Baseball teams, players, and game analysis."],
            10: ["Posts discussing the sport of ice hockey, including NHL teams, players, rules, and game analysis."],
            11: ["Posts discussing the science of secure communication through encoding information to prevent unauthorized access."],
            12: ["Posts discussing the design, construction, and analysis of electronic circuits and electrical devices."],
            13: ["Posts discussing human health, disease, medical treatments, and biomedical research findings."],
            14: ["Posts discussing outer space, celestial bodies, space missions, and the scientific study of the universe."],
            15: ["Posts discussing the beliefs, practices, scripture, and community life of the Christian religious tradition."],
            16: ["Posts discussing the legal, political, and social debates surrounding civilian ownership and regulation of firearms."],
            17: ["Posts discussing the political, military, and diplomatic affairs of countries in the Middle East region."],
            18: ["Posts discussing a broad range of political topics, ideologies, and current events not covered elsewhere."],
            19: ["Posts discussing spiritual beliefs, religious practices, and theological questions across various faith traditions."],
        },
    },
    
    # IMDB Movie Reviews - Binary sentiment
    "imdb": {
        "name_only": {
            0: ["neg"],
            1: ["pos"],
        },
        "description": {
            0: ["This text expresses negative sentiment, criticism, disappointment, or unfavorable opinion about a movie."],
            1: ["This text expresses positive sentiment, praise, enjoyment, or favorable opinion about a movie."],
        },
        "multi_description": {
            0: [
                "This movie review expresses negative sentiment or disappointment.",
                "Text criticizing a film, expressing dissatisfaction or poor opinion.",
                "A negative review indicating the movie was bad or not enjoyable.",
            ],
            1: [
                "This movie review expresses positive sentiment or praise.",
                "Text recommending a film, expressing enjoyment or high opinion.",
                "A positive review indicating the movie was good or enjoyable.",
            ],
        },
        "description_set_a": {
            0: ["A film review conveying unfavorable judgment, dissatisfaction, or critical disapproval of the cinematic work."],
            1: ["A film review conveying favorable judgment, enjoyment, or enthusiastic endorsement of the cinematic work."],
        },
        "description_set_b": {
            0: ["A review expressing an unfavorable affective response, characterized by displeasure, disappointment, or aversion."],
            1: ["A review expressing a favorable affective response, characterized by pleasure, satisfaction, or enthusiastic approval."],
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
        "multi_description": {
            0: [
                "This text expresses negative sentiment or a critical opinion.",
                "Content conveying dissatisfaction, criticism, or unfavorable view.",
                "A negative statement expressing disapproval or poor judgment.",
            ],
            1: [
                "This text expresses positive sentiment or a favorable opinion.",
                "Content conveying satisfaction, praise, or a positive view.",
                "A positive statement expressing approval or good judgment.",
            ],
        },
        "description_set_a": {
            0: ["A phrase or sentence conveying unfavorable evaluation, criticism, or emotional displeasure toward a subject."],
            1: ["A phrase or sentence conveying favorable evaluation, praise, or emotional satisfaction toward a subject."],
        },
        "description_set_b": {
            0: ["A linguistic expression reflecting an unpleasant subjective experience or unfavorable evaluative judgment."],
            1: ["A linguistic expression reflecting a pleasant subjective experience or favorable evaluative judgment."],
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
        "multi_description": {
            0: ["Text expressing admiration or deep respect.", "Content showing appreciation or high regard.", "Expression of admiration toward someone or something."],
            1: ["Text expressing amusement or finding something funny.", "Content showing humor, laughter, or entertainment.", "Expression of being amused or entertained."],
            2: ["Text expressing anger or strong displeasure.", "Content showing rage, fury, or hostility.", "Expression of anger or intense frustration."],
            3: ["Text expressing annoyance or mild irritation.", "Content showing frustration or being bothered.", "Expression of annoyance at something irritating."],
            4: ["Text expressing approval or agreement.", "Content showing endorsement or positive judgment.", "Expression of approving or agreeing with something."],
            5: ["Text expressing caring or compassion.", "Content showing concern, empathy, or desire to help.", "Expression of care or emotional support for others."],
            6: ["Text expressing confusion or lack of understanding.", "Content showing bewilderment or uncertainty.", "Expression of being confused or puzzled."],
            7: ["Text expressing curiosity or interest.", "Content showing inquisitiveness or desire to learn.", "Expression of curiosity about something."],
            8: ["Text expressing desire or longing.", "Content showing wanting, craving, or wishing.", "Expression of desire for something."],
            9: ["Text expressing disappointment or unmet expectations.", "Content showing letdown or dissatisfaction.", "Expression of being disappointed."],
            10: ["Text expressing disapproval or disagreement.", "Content showing rejection or negative judgment.", "Expression of disapproving of something."],
            11: ["Text expressing disgust or strong distaste.", "Content showing revulsion or repulsion.", "Expression of disgust toward something."],
            12: ["Text expressing embarrassment or shame.", "Content showing awkwardness or self-consciousness.", "Expression of feeling embarrassed."],
            13: ["Text expressing excitement or enthusiasm.", "Content showing eagerness or high energy.", "Expression of excitement or anticipation."],
            14: ["Text expressing fear or anxiety.", "Content showing worry, dread, or apprehension.", "Expression of being afraid or fearful."],
            15: ["Text expressing gratitude or thankfulness.", "Content showing appreciation for kindness.", "Expression of being grateful."],
            16: ["Text expressing grief or deep sorrow.", "Content showing mourning or profound sadness.", "Expression of grief over a loss."],
            17: ["Text expressing joy or happiness.", "Content showing delight, pleasure, or positive emotion.", "Expression of feeling joyful or happy."],
            18: ["Text expressing love or deep affection.", "Content showing fondness, caring, or romantic feelings.", "Expression of love toward someone."],
            19: ["Text expressing nervousness or anxiety.", "Content showing unease, jitters, or worried anticipation.", "Expression of feeling nervous."],
            20: ["Text expressing optimism or hopefulness.", "Content showing positive outlook or expectation.", "Expression of being optimistic."],
            21: ["Text expressing pride or accomplishment.", "Content showing satisfaction or positive self-regard.", "Expression of feeling proud."],
            22: ["Text expressing realization or sudden understanding.", "Content showing an epiphany or coming to awareness.", "Expression of realizing something."],
            23: ["Text expressing relief or release from worry.", "Content showing ease or comfort after tension.", "Expression of feeling relieved."],
            24: ["Text expressing remorse or regret.", "Content showing guilt or sorrow for wrongdoing.", "Expression of feeling remorseful."],
            25: ["Text expressing sadness or sorrow.", "Content showing unhappiness or emotional pain.", "Expression of feeling sad."],
            26: ["Text expressing surprise or astonishment.", "Content showing shock or unexpected reaction.", "Expression of being surprised."],
            27: ["Text expressing neutral emotion or no strong feeling.", "Content lacking clear emotional tone.", "Expression of a neutral or balanced emotional state."],
        },
        "description_set_a": {
            0: ["An emotional state characterized by high regard, reverence, or strong appreciation for another's qualities."],
            1: ["An emotional state characterized by finding something comical, entertaining, or pleasurably humorous."],
            2: ["An intense emotional state involving strong displeasure, hostility, or reactive aggression toward a perceived wrong."],
            3: ["A mild negative emotional state involving irritation, impatience, or low-level frustration with a stimulus."],
            4: ["An evaluative state involving agreement, endorsement, or positive judgment of an idea, action, or person."],
            5: ["An other-oriented emotional state involving concern, warmth, and motivation to support another's well-being."],
            6: ["A cognitive-emotional state involving uncertainty, disorientation, or inability to comprehend a situation clearly."],
            7: ["A motivational state involving interest, inquisitiveness, and desire to explore or acquire new information."],
            8: ["A motivational state involving longing, craving, or strong wish to obtain or experience something."],
            9: ["An emotional state arising when outcomes fall short of expectations, producing a sense of letdown."],
            10: ["An evaluative state involving rejection, objection, or negative judgment of an idea, action, or behavior."],
            11: ["An intense aversive emotional state involving revulsion, repugnance, or strong rejection of a stimulus."],
            12: ["A self-conscious emotional state arising from perceived social exposure, awkwardness, or loss of dignity."],
            13: ["A high-arousal positive emotional state involving enthusiasm, eager anticipation, and energized engagement."],
            14: ["A high-arousal negative emotional state involving perceived threat, danger, or anticipation of harm."],
            15: ["A positive social emotion involving acknowledgment of benefit received and appreciation toward a benefactor."],
            16: ["A profound negative emotional state involving deep sorrow, mourning, and distress following significant loss."],
            17: ["A high-arousal positive emotional state involving happiness, pleasure, and a sense of well-being."],
            18: ["A deep positive emotional state involving strong affection, attachment, and caring for another person."],
            19: ["A moderate-arousal negative state involving apprehension, unease, and worry about an uncertain outcome."],
            20: ["A positive future-oriented state involving confident expectation of favorable outcomes and hopeful anticipation."],
            21: ["A self-conscious positive emotion arising from achievement, competence, or association with valued outcomes."],
            22: ["A cognitive-emotional state involving sudden insight, comprehension, or awareness of a previously unclear truth."],
            23: ["A positive emotional state involving the reduction of tension, worry, or distress after a threat passes."],
            24: ["A self-conscious negative emotion involving guilt, regret, and sorrow for having caused harm or wrongdoing."],
            25: ["A low-arousal negative emotional state involving unhappiness, sorrow, and a sense of loss or misfortune."],
            26: ["A high-arousal emotional state triggered by an unexpected or unanticipated event, either positive or negative."],
            27: ["An absence of strong emotional valence, characterized by a calm, detached, or emotionally unengaged state."],
        },
        "description_set_b": {
            0: ["A positive evaluative emotion in which one perceives another as possessing qualities worthy of high esteem."],
            1: ["A positive emotion elicited by incongruity or absurdity, producing laughter or a sense of comic pleasure."],
            2: ["A primary emotion in Ekman's framework involving a strong aversive response to perceived injustice or obstruction."],
            3: ["A low-intensity negative emotion involving mild irritation or impatience in response to a minor frustration."],
            4: ["A positive evaluative response involving agreement with or endorsement of a person's actions or expressed views."],
            5: ["An other-directed positive emotion involving empathic concern and motivation to protect another's well-being."],
            6: ["A cognitive-affective state arising when incoming information cannot be integrated into existing mental schemas."],
            7: ["A positive epistemic emotion involving heightened interest and motivation to seek out new information or experiences."],
            8: ["A motivational state involving appetitive longing or craving directed toward an object, person, or outcome."],
            9: ["A negative emotion arising from the failure of an anticipated outcome to meet prior expectations."],
            10: ["A negative evaluative response involving moral or aesthetic objection to a person's actions or expressed views."],
            11: ["A primary emotion in Plutchik's wheel involving strong aversion or revulsion toward a perceived contaminant."],
            12: ["A self-conscious emotion arising from perceived public exposure of a personal failure or social transgression."],
            13: ["A high-arousal positive emotion involving energized anticipation and enthusiasm toward an upcoming event."],
            14: ["A primary emotion in Ekman's framework involving activation of threat-detection systems and avoidance motivation."],
            15: ["A positive social emotion involving recognition of benefit received from another and a desire to reciprocate."],
            16: ["A profound negative emotion involving prolonged distress and yearning following the loss of a valued attachment."],
            17: ["A primary positive emotion in Ekman's framework involving subjective well-being and hedonic pleasure."],
            18: ["A complex positive emotion involving deep attachment, warmth, and enduring affection toward another person."],
            19: ["A moderate-arousal negative emotion involving apprehension and somatic tension in anticipation of a threat."],
            20: ["A positive future-oriented emotion involving confident expectation of favorable outcomes and positive affect."],
            21: ["A self-conscious positive emotion arising from achievement of a valued goal or association with success."],
            22: ["A cognitive-affective state involving sudden comprehension or insight following a period of uncertainty."],
            23: ["A positive emotion involving the reduction of tension and restoration of equilibrium after a threat subsides."],
            24: ["A self-conscious negative emotion involving guilt and sorrow for having caused harm or violated moral standards."],
            25: ["A primary emotion in Ekman's framework involving low arousal, negative valence, and a sense of loss."],
            26: ["A primary emotion in Ekman's framework triggered by an unexpected stimulus, with positive or negative valence."],
            27: ["An affective baseline state characterized by the absence of strong emotional activation or valence."],
        },
    },
}


def get_label_texts(dataset_name: str, label_mode: str) -> Dict[int, List[str]]:
    """Get label texts for a dataset and mode.
    
    Supports both manual label definitions and LLM-generated descriptions.
    
    Args:
        dataset_name: Name of the dataset
        label_mode: Label mode (name_only, description, multi_description, l2, l3, etc.)
        
    Returns:
        Dictionary mapping label IDs to list of text representations
    """
    # Check for LLM-generated descriptions first (l2, l3, and variants)
    # l2 and l3 are anchored by default: "{label_name}: {description}"
    # l2_raw and l3_raw are non-anchored variants kept for ablation only
    if label_mode in ("l2", "l3", "l2_anchored", "l3_anchored", "l2_raw", "l3_raw"):
        # l2/l3/l2_anchored/l3_anchored are all anchored; l2_raw/l3_raw are not
        anchored = label_mode not in ("l2_raw", "l3_raw")
        base_mode = "l2" if label_mode in ("l2", "l2_anchored", "l2_raw") else "l3"

        generated = load_generated_descriptions(force_reload=False)
        if dataset_name in generated:
            # Get label names for anchoring (from name_only)
            name_only = LABEL_SETS.get(dataset_name, {}).get("name_only", {})

            result = {}
            for label_id, data in generated[dataset_name].items():
                label_id = int(label_id)
                if not isinstance(data, dict):
                    print(f"ERROR: Label {label_id} data is not a dict: {type(data)}")
                    continue
                if base_mode not in data:
                    print(f"ERROR: Label {label_id} missing {base_mode!r} key. Available: {list(data.keys())}")
                    continue

                label_name = name_only.get(label_id, [str(label_id)])[0] if name_only else str(label_id)

                if base_mode == "l2":
                    desc = data["l2"]
                    if anchored:
                        desc = f"{label_name}: {desc}"
                    result[label_id] = [desc]
                else:  # l3
                    descs = data["l3"]
                    if anchored:
                        descs = [f"{label_name}: {d}" for d in descs]
                    result[label_id] = descs
            return result
        else:
            raise ValueError(
                f"LLM-generated descriptions not found for dataset '{dataset_name}'. "
                f"Run 'python -m scripts.generate_label_descriptions --dataset {dataset_name}' first."
            )
    
    # Fall back to manual label definitions
    if dataset_name not in LABEL_SETS:
        raise ValueError(f"Dataset {dataset_name} not found in LABEL_SETS")
    
    if label_mode not in LABEL_SETS[dataset_name]:
        available_modes = list(LABEL_SETS[dataset_name].keys())
        raise ValueError(
            f"Label mode {label_mode} not found for dataset {dataset_name}. "
            f"Available modes: {available_modes}"
        )
    
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


def build_multi_description_embeddings(
    label_dict: Dict[int, List[str]],
    encoder,
    normalize: bool = True,
    batch_size: int = 32,
) -> tuple:
    """Build label embeddings for multi_description mode using mean pooling.

    Each label has multiple descriptions. Their embeddings are averaged (mean pooled)
    to produce a single representative embedding per class.

    Args:
        label_dict: Dict mapping label_id -> list of 3 description strings
        encoder: BiEncoder instance with .encode() method
        normalize: Whether to L2-normalize the final pooled embeddings
        batch_size: Batch size for encoding

    Returns:
        Tuple of (label_embeddings, label_ids)
        - label_embeddings: np.ndarray of shape (num_classes, dim)
        - label_ids: List[int] of class IDs in sorted order
    """
    import numpy as np

    label_embeddings = []
    label_ids = []

    for label_id in sorted(label_dict.keys()):
        descriptions = label_dict[label_id]
        if not descriptions:
            raise ValueError(f"Label {label_id} has no descriptions in multi_description mode.")

        # Encode all descriptions for this class: shape (k, dim)
        desc_embs = encoder.encode(
            descriptions,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=False,
            text_type="label",
        )
        desc_embs = np.asarray(desc_embs, dtype=np.float32)

        # Mean pool across descriptions
        pooled = desc_embs.mean(axis=0)  # shape (dim,)

        # Re-normalize after pooling
        if normalize:
            norm = np.linalg.norm(pooled)
            if norm > 1e-12:
                pooled = pooled / norm

        label_embeddings.append(pooled)
        label_ids.append(label_id)

    return np.stack(label_embeddings, axis=0), label_ids
