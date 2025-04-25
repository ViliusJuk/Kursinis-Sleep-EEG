import mne
import numpy as np

# Parametrai
edf_path = "sleep-edf/SC4001E0-PSG.edf"
hyp_path = "sleep-edf/SC4001EC-Hypnogram.edf"
segment_duration = 30  # sekundės

# Stadijų, kurias klasifikuosime, kodai
target_labels = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1
}

# Nuskaityti EEG ir anotacijas
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw.set_annotations(mne.read_annotations(hyp_path))

# Pasirinkti tik vieną EEG kanalą
raw.pick_channels(['EEG Fpz-Cz'])

# Anotacijas → Epochai (segmentai)
events, event_id = mne.events_from_annotations(raw, event_id=target_labels)

# Epochų kūrimas
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=0,
    tmax=segment_duration,
    baseline=None,
    detrend=1,
    preload=True
)

# Feature extraction funkcija
def extract_features(data, sfreq):
    data = data[0]  # shape: (channels, time)
    band_limits = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 49.0)]
    
    features = []
    for low_f, high_f in band_limits:
        filtered = mne.filter.filter_data(data, sfreq, low_f, high_f, verbose=False)
        energy = np.sum(filtered ** 2)
        entropy = -np.sum((filtered**2) * np.log(np.abs(filtered**2) + 1e-10))
        std = np.std(filtered)
        features.extend([energy, entropy, std])
    
    return features

# X ir y masyvų kūrimas
X, y = [], []
sfreq = epochs.info['sfreq']

for i in range(len(epochs)):
    epoch_data = epochs[i].get_data()
    label = epochs.events[i, 2]
    
    X.append(extract_features(epoch_data, sfreq))
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Iš viso segmentų: {len(X)}")
print(f"Požymių matricos forma: {X.shape}")
print(f"Žymės (y): {np.unique(y)}")


import pickle
import os

os.makedirs("src", exist_ok=True)

with open("src/Xy_data.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("Duomenys išsaugoti į Xy_data.pkl ✅")