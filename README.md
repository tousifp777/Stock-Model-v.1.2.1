# Hindi Stock Speech-to-Insight

Turn Hindi stock market conversations into actionable insights.  
This project uses transcription + translation + NLP to extract Buy/Sell calls, stock names, prices, and stop-loss levels from long financial videos or audios.


## Features

- 🎙️ Transcribe Hindi financial speech (AssemblyAI API)
- 🌐 Translate to English (Gemini API)
- 🔍 Extract stock names, price levels, buy/sell actions, stop-loss
- 📊 Export insights to structured CSV for analysis
- ⚡ Optimized pipeline for handling long hours of financial commentary

## Tech Stack

- Python
- AssemblyAI (Speech-to-Text)
- Gemini API (Translation)
- Pandas (Data processing)
- Regex/NLP (Entity extraction)
## Installation

- pip install -r Requirements.txt
## Usage

python main.py  ---->input your Video/Audio File

## Example Output

| Stock   | Price | Stop Loss | Action | Context Sentence |
|---------|-------|-----------|--------|------------------|
| MOIL    |       |           | Buy    | If you want to make some purchases, it's MOIL |
| MOIL    | 366   |           | Buy    | The recommendation is to buy MOIL, and the current closing price is around 366 |
| MOIL    | 400   | 400       | Buy    | So, the recommendation is to buy MOIL with a target of ₹400 and a stop loss of ₹350 |
| MOIL    |       |           | Buy    | So, a positional buying recommendation is coming for MOIL |
| GABRIEL |       |           | Buy    | So, I have a strong opinion on Gabriel India, and I believe you can stay invested in it, and you can definitely make fresh purchases |
| FORTIS  |       |           | Buy    | We have a buy recommendation on Fortis Healthcare |
| FORTIS  |       |           | Buy    | I believe you can buy Fortis Healthcare for the short term |
| FORTIS  |       | 774       |        | So, the recommendation for the short term is Fortis Healthcare, with a stop loss of 774 and a target of 800 rupees |




## Roadmap

- [ ] Improve Hindi-to-English translation accuracy
- [ ] Add support for multiple Indian languages
- [ ] Real-time stock call alerts
- [ ] Dashboard for visualization


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GEMINI_API_KEY`

`ASSEMBLYAI_API_KEY`


## Contributing

Contributions are welcome!  
Fork the repo, make changes, and submit a pull request.
