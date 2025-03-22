import mne
import matplotlib.pyplot as plt

# Nuskaitymas
raw = mne.io.read_raw_edf("sleep-edf/SC4001E0-PSG.edf", preload=True)

# Hipnogramos pridėjimas
annotations = mne.read_annotations("sleep-edf/SC4001EC-Hypnogram.edf")
raw.set_annotations(annotations)

# Vizualizacija
raw.plot(duration=60, n_channels=20, scalings='auto')

# Priverstinai parodo langą

plt.show()
