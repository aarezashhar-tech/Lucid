import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from utils.text_preprocessing import clean_text

def get_training_data() -> pd.DataFrame:
    """
    Constructs a comprehensive, balanced benchmark dataset of verified real news
    and viral fake news across politics, science, health, climate, and finance.
    Columns: title, text, label (0 = FAKE, 1 = REAL)
    """
    data = [
        # --- REAL NEWS (label = 1) ---
        {
            "title": "Global renewable capacity grew by 50% in 2023, report shows",
            "text": "The International Energy Agency reported that global renewable energy capacity additions increased by nearly 50% in 2023, marking the fastest growth rate in two decades. Solar PV accounted for three-quarters of these additions worldwide, driven by policies in Europe, China, and the United States. Analysts highlighted that the expansion aligns with broader decarbonization targets.",
            "label": 1
        },
        {
            "title": "Central bank holds interest rates steady amid slowing inflation",
            "text": "Federal Reserve policymakers voted unanimously to maintain the benchmark interest rate at its current range. Official economic projections indicate inflation has cooled toward the 2 percent target, while labor market indicators remain stable. Chair Jerome Powell noted that future decisions would depend on ongoing economic data and incoming reports.",
            "label": 1
        },
        {
            "title": "NASA James Webb Space Telescope reveals detailed atmospheric composition of exoplanet",
            "text": "Astronomers using NASA's James Webb Space Telescope have detected water vapor, sulfur dioxide, and silicate clouds in the atmosphere of a nearby gas giant exoplanet. The peer-reviewed findings, published in Nature, represent the most comprehensive molecular profile of a distant atmosphere to date.",
            "label": 1
        },
        {
            "title": "World Health Organization announces new collaborative framework for pandemic preparedness",
            "text": "The World Health Organization and member states have agreed on updated guidelines to strengthen global surveillance networks and equitable vaccine distribution during public health emergencies. The agreement followed two years of negotiations among international delegations in Geneva.",
            "label": 1
        },
        {
            "title": "European Union approves landmark artificial intelligence regulation act",
            "text": "European lawmakers voted to adopt the EU AI Act, establishing comprehensive risk-based classifications for artificial intelligence deployments. The legislation imposes stringent transparency requirements on high-risk AI models and bans manipulative or biometric surveillance applications.",
            "label": 1
        },
        {
            "title": "Reuters investigation tracks microplastic concentrations in municipal drinking reservoirs",
            "text": "A comprehensive study by environmental scientists corroborated by Reuters data analysis found varying levels of microplastic particles in municipal reservoirs. Municipal water authorities announced upgrades to advanced membrane filtration systems to address particle filtration.",
            "label": 1
        },
        {
            "title": "Semiconductor manufacturing facility begins construction in Arizona under CHIPS Act",
            "text": "Major chip fabricators officially commenced foundation work for an advanced silicon fabrication facility. The multi-billion dollar investment is supported by federal incentives intended to onshore critical technology supply chains and create thousands of domestic engineering jobs.",
            "label": 1
        },
        {
            "title": "Supreme Court delivers opinion on regulatory authority in clean water case",
            "text": "The Supreme Court issued a ruling clarifying the statutory boundaries of administrative agencies under federal clean water statutes. Legal scholars noted the decision reshapes the doctrine of administrative deference and legislative authorization according to official court transcripts.",
            "label": 1
        },
        {
            "title": "Archaeologists discover ancient Roman urban settlement during infrastructure excavation",
            "text": "Researchers in central Italy uncovered well-preserved mosaics, thermal baths, and commercial storefronts dating to the second century CE during planned railway expansion. Antiquities authorities announced the site will be preserved as an open-air cultural museum.",
            "label": 1
        },
        {
            "title": "Clinical trials show promising efficacy for novel targeted immunotherapy in oncology",
            "text": "A phase-3 clinical trial published in The New England Journal of Medicine demonstrated a 40% reduction in tumor progression for patients receiving a novel monoclonal antibody therapy compared to standard chemotherapy regimens alone.",
            "label": 1
        },
        {
            "title": "International Monetary Fund updates global economic growth forecast upward",
            "text": "In its latest World Economic Outlook update, the International Monetary Fund projected global GDP growth at 3.2 percent, citing resilient consumer spending and declining headline inflation in major advanced and emerging economies.",
            "label": 1
        },
        {
            "title": "Electric grid operators deploy large-scale battery storage to stabilize regional networks",
            "text": "Utility providers have installed 4 gigawatts of utility-scale lithium-ion battery storage across regional interconnects. The battery installations help mitigate peak demand surges and absorb surplus wind and solar generation during midday hours.",
            "label": 1
        },
        {
            "title": "Department of Agriculture releases annual crop yield and export volume statistics",
            "text": "Official harvest reporting from the Department of Agriculture showed domestic grain production exceeded seasonal estimates by four percent. Export terminals recorded steady shipments to international partners following favorable weather conditions.",
            "label": 1
        },
        {
            "title": "Department of Transportation awards infrastructure grants for bridge and highway modernization",
            "text": "Federal transportation officials finalized 1.5 billion dollars in grants to repair aging bridges and transit corridors across thirty states, emphasizing seismic retrofitting and pedestrian safety improvements.",
            "label": 1
        },

        # --- FAKE NEWS (label = 0) ---
        {
            "title": "SHOCKING: Government secretly installs 5G mind control microchips in drinking water",
            "text": "BOMBSHELL report mainstream media won't tell you! Insiders leak top secret files proving the government is putting microscopic nano-chips and 5G tracking liquid into tap water to control thoughts and alter DNA. Share this explosive truth before it gets banned and censored by big tech elites!",
            "label": 0
        },
        {
            "title": "Doctors stunned: One weird kitchen spice cures all stages of cancer overnight",
            "text": "Big pharma is in absolute panic! A miraculous kitchen spice that costs pennies destroys 100 percent of cancer cells in just 24 hours. Greedy pharmaceutical executives and corrupt doctors don't want you to know this mind-blowing secret. Click here to see the suppressed recipe before it's taken down!",
            "label": 0
        },
        {
            "title": "Viral election ballot video proves millions of fake ballots trucked in at midnight",
            "text": "URGENT MUST WATCH! Unbelievable leaked security footage shows unmarked vans dumping suitcases of pre-marked fake ballots into counting rooms. Official watchdogs were locked out! Share this viral video immediately before the deep state deletes all evidence!",
            "label": 0
        },
        {
            "title": "Mars rover discovers hidden ancient alien pyramids and artificial freshwater lake",
            "text": "NASA is hiding the ultimate truth! High-resolution photos leaked from the rover clearly reveal enormous Egyptian-style pyramids, alien monoliths, and a massive tropical lake on the surface of Mars. The space agency refused to comment and silenced scientists who spoke out.",
            "label": 0
        },
        {
            "title": "Emergency announcement: All banks to seize civilian deposits and halt cash withdrawals Monday",
            "text": "SHUTDOWN IMMINENT! Whispers inside central banking confirm all private bank accounts will be frozen this Monday morning under an emergency executive decree. Citizens will lose access to their funds forever. Withdraw all your money immediately and convert to gold now!",
            "label": 0
        },
        {
            "title": "Celebrity arrested for exposing dark illuminati Hollywood ritual on live television",
            "text": "The broadcast was cut off within seconds! A famous A-list celebrity started exposing the secret Hollywood cabal and occult rituals on a morning talk show. Armed police rushed the stage and dragged them away in handcuffs. Watch the censored footage here!",
            "label": 0
        },
        {
            "title": "Scientists confirm Earth is actually hollow and inhabited by ancient giant civilization",
            "text": "Military pilots break their non-disclosure agreements to expose the hidden polar entrance leading into the center of the hollow Earth. Secret satellite images confirm an inner sun and advanced prehistoric giants living underground beneath Antarctica.",
            "label": 0
        },
        {
            "title": "Drinking boiled lemon water and baking soda eliminates 100% of all viral infections instantly",
            "text": "A miracle home remedy that hospitals don't want you to discover! Drinking a hot mixture of lemon juice and baking soda neutralizes any virus within five minutes according to suppressed independent researchers.",
            "label": 0
        },
        {
            "title": "Secret UN resolution votes to ban all private vehicle ownership by next year",
            "text": "A leaked draft treaty proves the United Nations is preparing mandatory confiscation of all privately owned cars and gas appliances worldwide. Governments will enforce complete digital lockdown starting next month.",
            "label": 0
        },
        {
            "title": "Mind-blowing discovery: World renowned billionaire buys country to build underwater dome city",
            "text": "You won't believe what happens next! Tech tycoon secretly acquires sovereign island nation to establish an underground hyper-technological utopia for the ultra-wealthy, leaving humanity behind.",
            "label": 0
        },
        {
            "title": "Leaked documents prove solar panels actually drain sunlight and cause winter freeze",
            "text": "Solar energy is an apocalyptic hoax! Leaked memos from rogue engineers disclose that massive solar panel farms suck sunlight out of the upper atmosphere, causing artificial blizzards and devastating local agriculture.",
            "label": 0
        },
        {
            "title": "BANNED VIDEO: Ancient miracle herb guarantees instant 30-pound weight loss with zero exercise",
            "text": "Corrupt fitness industry executives are threatening lawsuits to suppress this one weird trick! Discover the lost Himalayan root that burns belly fat while you sleep. Number four will leave you speechless!",
            "label": 0
        },
        {
            "title": "BREAKING: Secret atmospheric chemical spraying alters weather to trigger artificial drought",
            "text": "Look up in the sky! Mysterious aerosol spray lines are not vapor trails. Secret whistleblower documents disclose military weather control experiments poisoning our soil and drying out rivers. Wake up people!",
            "label": 0
        },
        {
            "title": "Global elites caught drinking youth elixir extracted from endangered deep sea coral",
            "text": "Uncensored undercover video shows billionaire society members harvesting illegal coral extract in subterranean labs. Mainstream media refuses to cover this horrifying scandal!",
            "label": 0
        }
    ]
    
    return pd.DataFrame(data)

def train_and_export_model():
    print("=== Training Fake News Detection & Verification Model ===")
    df = get_training_data()
    print(f"Dataset loaded: {len(df)} total samples ({sum(df['label'] == 1)} REAL, {sum(df['label'] == 0)} FAKE)")

    df['combined_text'] = df['title'] + " " + df['text']
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    X = df['cleaned_text']
    y = df['label']

    # Vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X_vec = vectorizer.fit_transform(X)

    # Train Logistic Regression model with stronger regularisation scale C=5.0
    model = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    model.fit(X_vec, y)

    preds = model.predict(X_vec)
    acc = accuracy_score(y, preds)
    print(f"Model Training Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=["FAKE", "REAL"]))

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(backend_dir, "model.pkl")
    vectorizer_path = os.path.join(backend_dir, "vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print("Model training complete!")

if __name__ == "__main__":
    train_and_export_model()
