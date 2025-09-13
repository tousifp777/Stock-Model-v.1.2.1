import assemblyai as aai
from tkinter import Tk, filedialog
import pandas as pd
import re
import os
from typing import List, Dict, Tuple
from rapidfuzz import process, fuzz
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# =========================
# Config
# =========================
'''
Insert your API Keys at, 
line 28 and line 215
'''
STOCK_LIST_FILE = "EQUITY_L.csv"
TRANSCRIPT_TXT = "Transcript_output.txt"
TRANSLATED_TXT = "Transcript_translated.txt"
OUTPUT_CSV = "Analysis.csv"

# =========================
# Translator
# =========================
client = genai.Client(api_key= os.getenv('GEMINI_API_KEY'))   #------> GEMINI_API_KEY = YOUR API KEY OF GOOGLE GEMINI

def chunk_text(text, max_words=2000):
    """Split text into word chunks for Gemini API."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def translate_text(text: str) -> str:
    """Translate Hindi text to English using Gemini."""
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a professional translation assistance which translate hindi transcription into English. The transcription contain hindi news and Stock information."),
        contents=text
    )
    return response.text.strip()

def translate_file(input_file=TRANSCRIPT_TXT, output_file=TRANSLATED_TXT):
    with open(input_file, "r", encoding="utf-8") as f:
        hindi_text = f.read()

    print("Splitting transcript into chunks for translation...")
    chunks = list(chunk_text(hindi_text))

    translated_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Translating chunk {i}/{len(chunks)} ...")
        translated = translate_text(chunk)
        translated_chunks.append(translated)

    final_text = "\n".join(translated_chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"Translation complete. Saved to {output_file}")
    return final_text

# =========================
# Utilities
# =========================
def load_company_list(csv_path: str) -> Tuple[pd.DataFrame, set, set]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    name_col = None
    symbol_col = None
    for k, v in cols.items():
        if k in ("name of company", "company name", "name", "company", "security name"):
            name_col = v
        if k in ("symbol", "ticker", "code", "security code"):
            symbol_col = v

    if name_col is None and symbol_col is None:
        raise ValueError("Could not find a company name or symbol column in EQUITY_L.csv")

    names = set()
    symbols = set()

    if name_col:
        names = set(
            df[name_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

    if symbol_col:
        symbols = set(
            df[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
    return df, names, symbols


def split_sentences(text: str) -> List[str]:
    """Split English sentences."""
    text = re.sub(r'\s+', ' ', text)
    parts = re.split(r'[.?!]+', text)
    return [p.strip() for p in parts if p.strip()]


BUY_WORDS = ["buy", "accumulate", "go long", "purchase"]
SELL_WORDS = ["sell", "exit", "book profit", "go short"]
STOPLOSS_WORDS = ["stop loss", "sl"]

NUM = r'[0-9][\d,]*\.?\d*'
PRICE_PAT = re.compile(rf'(?:₹|rs\.?|rupees?)\s*({NUM})', flags=re.IGNORECASE)
STANDALONE_NUM = re.compile(rf'\b({NUM})\b')

def find_action(sentence: str) -> str:
    s = sentence.lower()
    if any(w in s for w in SELL_WORDS):
        return "Sell"
    if any(w in s for w in BUY_WORDS):
        return "Buy"
    return ""

def extract_stoploss(sentence: str) -> str:
    s = sentence.lower()
    for kw in STOPLOSS_WORDS:
        if kw in s:
            m = PRICE_PAT.search(s)
            if m:
                return m.group(1).replace(",", "")
            m2 = STANDALONE_NUM.search(s)
            if m2:
                return m2.group(1).replace(",", "")
    return ""

def extract_price_near_action(sentence: str, action: str) -> str:
    if not action:
        return ""
    m = PRICE_PAT.search(sentence)
    if m:
        return m.group(1).replace(",", "")
    for m2 in STANDALONE_NUM.finditer(sentence):
        num = m2.group(1)
        try:
            val = float(num.replace(",", ""))
            if 5 <= val <= 1000000:
                return num.replace(",", "")
        except:
            pass
    return ""

def pick_company_match(sentence: str, names_upper: set, symbols_upper: set) -> str:
    sentence_upper = sentence.upper()

    # Company name match
    found_name = ""
    max_len = 0
    for nm in names_upper:
        if nm and nm in sentence_upper:
            if len(nm) > max_len:
                found_name, max_len = nm, len(nm)
    if found_name:
        return found_name

    # Symbol match
    for sym in symbols_upper:
        if sym and re.search(rf'\b{re.escape(sym)}\b', sentence_upper):
            return sym

    return ""

def analyze_transcript_to_rows(transcript_text: str, names_upper: set, symbols_upper: set) -> List[Dict]:
    rows = []
    sentences = split_sentences(transcript_text)
    for sent in sentences:
        company = pick_company_match(sent, names_upper, symbols_upper)
        if not company:
            continue
        action = find_action(sent)
        price = extract_price_near_action(sent, action)
        stoploss = extract_stoploss(sent)
        if action or price or stoploss:
            rows.append({
                "Stock Name": company,
                "Stock Price": price,
                "Stop Loss": stoploss,
                "Action": action,
                "Context Sentence": sent
            })
    return rows

# =========================
# Main pipeline
# =========================
def main():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select your video/audio file",
        filetypes=[("Audio/Video Files", "*.mp3 *.mp4 *.wav *.m4a *.mov")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    # Transcribe
    aai.settings.api_key = os.getenv('ASSEMBLY_API_KEY') #-------> ASSEMBLY_API_KEY = YOUR ASSEMBLY AI API KEY
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        language_detection=True
    )
    print(f"Transcribing: {file_path} ...")
    transcript = aai.Transcriber(config=config).transcribe(file_path)
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f:
        f.write(transcript.text)
    print(f"Transcription saved to {TRANSCRIPT_TXT}")

    # Translate Hindi → English
    english_text = translate_file()

    # Load stock list
    if not os.path.exists(STOCK_LIST_FILE):
        raise FileNotFoundError(f"{STOCK_LIST_FILE} not found in current directory.")
    _, names_upper, symbols_upper = load_company_list(STOCK_LIST_FILE)

    # Analyze on English text
    rows = analyze_transcript_to_rows(english_text, names_upper, symbols_upper)

    # Save CSV
    if rows:
        df = pd.DataFrame(rows, columns=["Stock Name", "Stock Price", "Stop Loss", "Action", "Context Sentence"])
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Analysis saved to {OUTPUT_CSV}")
    else:
        print("No actionable stock mentions found. Check transcript or keyword list.")

if __name__ == "__main__":
    main()
