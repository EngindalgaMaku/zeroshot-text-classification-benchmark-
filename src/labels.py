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
