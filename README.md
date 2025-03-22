# Klasifikavimo algoritmų tyrimas naudojant miego EEG duomenis

Šis projektas yra Vilniaus universiteto informacinių technologijų studijų kursinis darbas. Tikslas – ištirti miego EEG duomenis ir pritaikyti klasifikavimo algoritmus miego stadijų atpažinimui.

## 🧠 Pagrindiniai žingsniai

- Sleep-EDF [Expanded] duomenų rinkinio analizė
- `.edf` failų nuskaitymas su `MNE` biblioteka
- Miego stadijų anotacijų naudojimas klasifikacijai
- EEG signalų filtravimas į sub-bands
- Features (energija, entropija, std) išgavimas
- Klasifikavimas naudojant SVM

## 💻 Naudojamos technologijos

- Python 3.9
- `mne`
- `pyedflib`
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`

## 📁 Projekto struktūra

```
kursinis2025/
├── sleep-edf/              # Parsisiųsti EEG + hipnogramos failai
│   ├── SC4001E0-PSG.edf
│   └── SC4001EC-Hypnogram.edf
├── download.py             # Duomenų parsisiuntimo ir nuskaitymo skriptas
├── vizualizacija.py        # EEG vizualizacija
├── .gitignore
├── requirements.txt
└── README.md
```
