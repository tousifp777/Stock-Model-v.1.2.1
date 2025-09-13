# translator.py
from google import genai
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()

# Configure Gemini
genai.configure(api_key="GEMINI_API_KEY")  # replace with your key
model = genai.GenerativeModel("gemini-2.5-flash")

def chunk_text(text, max_words=2000):
    """Split long text into chunks of max_words size."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def translate_text(text):
    """Translate Hindi text to English using Gemini."""

    prompt = f"You are a professional translation assistance which translate hindi transcription into English. The trnscription contain hindi news and Stock information: \n\n{text}"

    response = model.generate_content(prompt)
    
    return response.text.strip()

def translate_file(input_file="transcript_output.txt", output_file="transcript_translated.txt"):
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


if __name__ == "__main__":
    configure()
    translate_file()
