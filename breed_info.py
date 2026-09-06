# breed_info.py
"""
Metadata database for Indian Cattle and Buffalo breeds.
Used by app.py to enrich the model's raw class prediction with
human-readable breed information shown on the Prediction Result page.

Feel free to add/edit breeds here — this dictionary key MUST exactly
match the folder name (class name) used while training the model
(see dataset/README.md and train_model.py).
"""

BREED_INFO = {
    # ----------------------------- CATTLE -----------------------------
    "Gir": {
        "species": "Cattle",
        "origin": "Gujarat, India",
        "uses": ["Dairy production", "Drought resistance", "Heat tolerance", "Indigenous breed conservation"],
        "characteristics": ["Medium to large size", "Distinctive curved horns", "Long, pendulous ears",
                             "Prominent forehead and strong body", "Reddish coat with white patches", "Good milch animals"],
        "description": "Gir is one of the most important indigenous cattle breeds of India. It is well known for "
                       "its high milk production and adaptability to harsh climate conditions. It has a calm "
                       "nature and is widely reared for dairy purposes."
    },
    "Sahiwal": {
        "species": "Cattle",
        "origin": "Punjab region (India-Pakistan border)",
        "uses": ["Dairy production", "Crossbreeding for milk improvement", "Draught in some regions"],
        "characteristics": ["Reddish-brown coat", "Loose skin", "Short horns", "Heat and tick resistant",
                             "Docile temperament"],
        "description": "Sahiwal is one of the best indigenous dairy breeds of the Indian subcontinent, valued "
                       "for high milk yield, heat tolerance, and disease resistance. It is widely used to "
                       "improve local cattle through crossbreeding programs."
    },
    "Red Sindhi": {
        "species": "Cattle",
        "origin": "Sindh region (now Pakistan), reared across India",
        "uses": ["Dairy production", "Heat and disease resistance breeding programs"],
        "characteristics": ["Deep red coat", "Compact and well-proportioned body", "Short curved horns",
                             "Good heat tolerance"],
        "description": "Red Sindhi is a hardy dairy breed known for consistent milk yield under tropical "
                       "conditions and strong resistance to diseases and parasites."
    },
    "Tharparkar": {
        "species": "Cattle",
        "origin": "Rajasthan (Thar Desert), India",
        "uses": ["Dual purpose - milk and draught", "Arid zone farming"],
        "characteristics": ["White or light grey coat", "Medium size", "Lyre-shaped horns",
                             "High drought tolerance"],
        "description": "Tharparkar is a dual-purpose breed adapted to the arid Thar Desert region, valued for "
                       "both milk production and draught power under extreme heat and water scarcity."
    },
    "Rathi": {
        "species": "Cattle",
        "origin": "Rajasthan, India",
        "uses": ["Dairy production", "Suited to arid regions"],
        "characteristics": ["Brown and white patched coat", "Medium size", "Well-developed udder",
                             "Good heat tolerance"],
        "description": "Rathi cattle are recognized for good milk yield relative to body size and their "
                       "ability to survive on sparse desert fodder."
    },
    "Kankrej": {
        "species": "Cattle",
        "origin": "Gujarat & Rajasthan border, India",
        "uses": ["Dual purpose - milk and draught", "Fast, strong draught animal"],
        "characteristics": ["Silver-grey to steel-grey coat", "Large lyre-shaped horns", "Muscular body",
                             "Strong hooves"],
        "description": "Kankrej is a powerful dual-purpose breed prized for its strength as a draught animal "
                       "as well as reasonable milk yield, and is one of the ancestors of American Brahman cattle."
    },
    "Ongole": {
        "species": "Cattle",
        "origin": "Andhra Pradesh, India",
        "uses": ["Draught power", "Export/breeding stock (basis of Brahman breed)"],
        "characteristics": ["White or light grey coat", "Large hump", "Heavy dewlap", "Massive body size"],
        "description": "Ongole is a large, muscular breed known for strength and endurance. It has been "
                       "exported worldwide and used to develop breeds such as the American Brahman."
    },
    "Hariana": {
        "species": "Cattle",
        "origin": "Haryana, India",
        "uses": ["Dual purpose - milk and draught"],
        "characteristics": ["White or light grey coat", "Medium-sized horns", "Compact body",
                             "Good working ability"],
        "description": "Hariana is a hardy dual-purpose breed traditionally used for both milk and farm "
                       "work across northern India."
    },
    "Deoni": {
        "species": "Cattle",
        "origin": "Maharashtra & Karnataka, India",
        "uses": ["Dual purpose - milk and draught"],
        "characteristics": ["Black and white spotted coat", "Medium build", "Curved horns"],
        "description": "Deoni cattle are a spotted dual-purpose breed valued for moderate milk yield and "
                       "reliable draught performance in the Deccan plateau region."
    },
    "Khillari": {
        "species": "Cattle",
        "origin": "Maharashtra & Karnataka, India",
        "uses": ["Draught power", "Fast bullock cart racing breed"],
        "characteristics": ["Grey/white coat", "Compact muscular body", "Long horns", "Very fast and agile"],
        "description": "Khillari is a draught breed known for speed and stamina, traditionally used for "
                       "bullock-cart transport and racing."
    },
    "Kangayam": {
        "species": "Cattle",
        "origin": "Tamil Nadu, India",
        "uses": ["Draught power", "Jallikattu (bull-taming sport)"],
        "characteristics": ["White/grey coat with dark markings on head", "Muscular body", "Sturdy legs"],
        "description": "Kangayam cattle are a hardy draught breed from Tamil Nadu, well known for their "
                       "role in traditional Jallikattu events and farm work."
    },
    "Vechur": {
        "species": "Cattle",
        "origin": "Kerala, India",
        "uses": ["Dairy (low input, backyard rearing)", "Genetic conservation"],
        "characteristics": ["Very small size (one of the smallest cattle breeds)", "Red/brown or black coat",
                             "Small curved horns"],
        "description": "Vechur is a diminutive indigenous breed from Kerala, valued for its low feed "
                       "requirement, disease resistance, and high butterfat milk despite its small size."
    },
    "Punganur": {
        "species": "Cattle",
        "origin": "Andhra Pradesh, India",
        "uses": ["Dairy (low input)", "Genetic conservation"],
        "characteristics": ["Extremely small stature", "White/grey coat", "Short legs"],
        "description": "Punganur is among the smallest cattle breeds in the world, hardy and well suited to "
                       "backyard rearing with modest feed requirements."
    },
    "Amritmahal": {
        "species": "Cattle",
        "origin": "Karnataka, India",
        "uses": ["Draught power (military/transport history)"],
        "characteristics": ["Grey coat", "Long horns", "Lean muscular build", "High stamina"],
        "description": "Amritmahal cattle were historically bred for military transport and are known for "
                       "endurance and agility as draught animals."
    },
    "Hallikar": {
        "species": "Cattle",
        "origin": "Karnataka, India",
        "uses": ["Draught power", "Bullock cart racing"],
        "characteristics": ["Grey/white coat", "Long backward-curving horns", "Compact strong body"],
        "description": "Hallikar is a renowned draught breed known for its speed, strength, and distinctive "
                       "long horns, forming the base stock for several other South Indian breeds."
    },
    "Nagori": {
        "species": "Cattle",
        "origin": "Rajasthan, India",
        "uses": ["Draught power (fast trotting bullocks)"],
        "characteristics": ["White coat", "Tall elegant body", "Small horns"],
        "description": "Nagori cattle are prized as fast, elegant trotting bullocks used traditionally for "
                       "carts and light draught work in Rajasthan."
    },
    "Malvi": {
        "species": "Cattle",
        "origin": "Madhya Pradesh, India",
        "uses": ["Draught power"],
        "characteristics": ["Grey/white coat", "Black markings on head, neck and hump", "Sturdy build"],
        "description": "Malvi is a draught breed from the Malwa plateau, valued for its hardiness and "
                       "working capacity in agricultural operations."
    },
    "Nimari": {
        "species": "Cattle",
        "origin": "Madhya Pradesh, India",
        "uses": ["Dual purpose - milk and draught"],
        "characteristics": ["Red/brown coat with white patches", "Medium horns", "Well-built frame"],
        "description": "Nimari cattle from the Narmada valley are used for both moderate milk production and "
                       "farm work, adapted to the local climate."
    },
    "Dangi": {
        "species": "Cattle",
        "origin": "Maharashtra, India",
        "uses": ["Draught power (rice-growing tracts)"],
        "characteristics": ["Red and white or black and white spotted coat", "Short legs", "Sturdy hooves"],
        "description": "Dangi cattle are adapted to heavy rainfall and waterlogged rice-growing regions, "
                       "prized for their sure-footedness in muddy fields."
    },
    "Krishna Valley": {
        "species": "Cattle",
        "origin": "Karnataka & Maharashtra, India",
        "uses": ["Draught power (heavy soil ploughing)"],
        "characteristics": ["Grey/white coat", "Heavy massive body", "Short thick horns"],
        "description": "Krishna Valley cattle are large, powerful draught animals traditionally used for "
                       "ploughing the black cotton soils of the Krishna river basin."
    },

    # ---------------------------- BUFFALOES ----------------------------
    "Murrah": {
        "species": "Buffalo",
        "origin": "Haryana & Punjab, India",
        "uses": ["Dairy production (highest milk yielding buffalo breed)", "Crossbreeding to improve local buffaloes"],
        "characteristics": ["Jet black, glossy coat", "Tightly curled horns", "Massive well-developed udder",
                             "Broad body"],
        "description": "Murrah is the most popular and highest-yielding dairy buffalo breed in India, widely "
                       "used across the country and internationally for milk production and breeding programs."
    },
    "Jaffarabadi": {
        "species": "Buffalo",
        "origin": "Gujarat (Gir & Kutch region), India",
        "uses": ["Dairy production", "Draught in some areas"],
        "characteristics": ["Massive, heavy body", "Broad drooping horns curling near the neck",
                             "Black coat", "Large head"],
        "description": "Jaffarabadi is the heaviest Indian buffalo breed, known for high milk fat content and "
                       "well suited to marshy and coastal grazing areas of Gujarat."
    },
    "Mehsana": {
        "species": "Buffalo",
        "origin": "Gujarat, India",
        "uses": ["Dairy production"],
        "characteristics": ["Black or brownish-black coat", "Sickle-shaped drooping horns", "Medium-large body"],
        "description": "Mehsana buffaloes, derived from Murrah and Surti crosses, are a productive dairy "
                       "breed common in North Gujarat."
    },
    "Surti": {
        "species": "Buffalo",
        "origin": "Gujarat, India",
        "uses": ["Dairy production (high fat content milk)"],
        "characteristics": ["Brown to black coat", "Sickle-shaped flat horns", "Medium-sized compact body"],
        "description": "Surti buffaloes are known for milk with very high fat content, making them valuable "
                       "for ghee and dairy product manufacturing."
    },
    "Nili-Ravi": {
        "species": "Buffalo",
        "origin": "Punjab region (India-Pakistan)",
        "uses": ["Dairy production (very high yield)"],
        "characteristics": ["Jet black coat", "Small tightly curled horns", "Often has wall (blue) eyes",
                             "White markings on face/legs/tail"],
        "description": "Nili-Ravi is one of the best dairy buffalo breeds in the world, recognized by its "
                       "distinctive walled eyes and excellent milk production."
    },
    "Bhadawari": {
        "species": "Buffalo",
        "origin": "Uttar Pradesh & Madhya Pradesh, India",
        "uses": ["Dairy production (high fat milk)", "Draught power"],
        "characteristics": ["Copper/light brown coat", "Small compact body", "Curved horns"],
        "description": "Bhadawari buffaloes produce milk with exceptionally high fat content and are well "
                       "adapted to marshy, ravine terrain of the Yamuna-Chambal region."
    },
    "Nagpuri": {
        "species": "Buffalo",
        "origin": "Maharashtra, India",
        "uses": ["Dairy production", "Draught power"],
        "characteristics": ["Black coat, often with white markings", "Long flat sword-like horns",
                             "Medium-sized body"],
        "description": "Nagpuri buffaloes are a dual-purpose breed from Vidarbha, valued for moderate milk "
                       "yield and draught use in agriculture."
    },
    "Toda": {
        "species": "Buffalo",
        "origin": "Nilgiris, Tamil Nadu, India",
        "uses": ["Dairy (traditional/tribal rearing)", "Genetic conservation"],
        "characteristics": ["Curly long coat", "Large curved horns", "Sturdy hill-adapted body"],
        "description": "Toda buffaloes are a rare semi-wild breed reared by the Toda tribal community in the "
                       "Nilgiri hills, adapted to cold, high-altitude grasslands."
    },
    "Pandharpuri": {
        "species": "Buffalo",
        "origin": "Maharashtra, India",
        "uses": ["Dairy production", "Draught power"],
        "characteristics": ["Black coat", "Very long backward-spiraling horns", "Lean, long body"],
        "description": "Pandharpuri buffaloes are recognized by their unusually long, spiral horns and are "
                       "reared for both milk and farm work in southern Maharashtra."
    },
    "Banni": {
        "species": "Buffalo",
        "origin": "Kutch, Gujarat, India",
        "uses": ["Dairy production (arid/saline tolerant)"],
        "characteristics": ["Black coat", "Curled horns", "Compact hardy body", "Excellent grazing ability"],
        "description": "Banni buffaloes are hardy animals from the Banni grasslands of Kutch, capable of "
                       "surviving on saline water and sparse vegetation while maintaining good milk yield."
    },
}

# Convenience list of class names, in a fixed sorted order.
# IMPORTANT: this order should match the order used during model training
# (train_model.py saves the true mapping to model/class_indices.json,
# which app.py prefers over this list whenever it is present).
CLASS_NAMES = sorted(BREED_INFO.keys())

DEFAULT_INFO = {
    "species": "Unknown",
    "origin": "Not available",
    "uses": ["Information not available for this breed yet."],
    "characteristics": ["Information not available for this breed yet."],
    "description": "No description available. Add an entry for this breed in breed_info.py."
}


def get_breed_info(breed_name: str) -> dict:
    """Return metadata for a breed, falling back to a safe default."""
    return BREED_INFO.get(breed_name, DEFAULT_INFO)
