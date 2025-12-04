# Manga Translator
Locally working (offline) Manga translator JP->EN

## Features :

* **Single snipping (Shift + Q)** Manually select text to translate ; recommended for not detected bubbles or text outside of them. Can be also used as text extractor
* **Live translations(Shift +E)** Select area that you want model to automatically find bubbles in , output can be seen in GUI
* **Runs fully locally** First run may be slower due to fetching models and downloading them but once , later launches should be faster
*   Hotkeys are changeable

## Preview:
![preview](https://github.com/user-attachments/assets/cbb2bb8c-e658-48d9-877a-be26c6add4d1)


## Instalation

For development **Python3.10** was used thats why I encourage you to use the same version to avoid any complications  
**Pytorch 2.6.1 with CUDA** is also recommended for faster translations

```bash
# Clone the repo
git clone [https://github.com/F-iol/Manga-Translator.git](https://github.com/F-iol/Manga-Translator.git)
cd Manga-Translator

# Create and activate venv (Recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```
## Usage
Either open main.py file or run
```bash
python main.py
```

## Known issues :
- When using Shft+Q (single snipping option) sometimes old displayed images turn white
- When trying to translate a lot of text at once translation model either hallucinate or doesnt provide any good translations (fine tunning model would fix an issue)
- In live mode when bubbles are very close to each other they may be detected as one big bubble, as long as there isnt a lot of text shouldn't be an issue

## License
This project is licensed under the GNU GPLv3. You are free to use and modify it, but if you distribute modifications, they must remain open-source
