import os
import wget
import mne

# 1. Sukuriam katalogą duomenims
os.makedirs("sleep-edf", exist_ok=True)

# 2. Nurodom failų nuorodas
urls = [
    "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf?download",
    "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf?download"
]

# 3. Parsisiunčiam abu failus
for url in urls:
    print(f"Atsiunčiam: {url}")
    wget.download(url, out="sleep-edf/")
print("\n✅ Failai sėkmingai parsiųsti.\n")

# 4. Nuskaitymas su MNE
raw = mne.io.read_raw_edf("sleep-edf/SC4001E0-PSG.edf", preload=True)
annotations = mne.read_annotations("sleep-edf/SC4001EC-Hypnogram.edf")
raw.set_annotations(annotations)

# 5. Atspausdinam info
print(raw.info)
print("\n📌 Anotacijos:")
print(annotations)

# 6. Vizualizacija (atsidarys interaktyvus langas)
raw.plot(duration=60, n_channels=20, scalings='auto')
