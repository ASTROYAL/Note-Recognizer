import pyaudio
import numpy as np
import librosa

# Constants
CHUNK = 2048
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

# Guitar tuning (standard EADGBE)
guitar_strings = {
    'E2': 82.41,
    'A2': 110.00,
    'D3': 146.83,
    'G3': 196.00,
    'B3': 246.94,
    'E4': 329.63
}

def closest_note_frequency(frequency):
    min_diff = float('inf')
    closest_note = None
    for note, freq in guitar_strings.items():
        diff = abs(frequency - freq)
        if diff < min_diff:
            min_diff = diff
            closest_note = note
    return closest_note

def determine_string_and_fret_harmonics(data):
    # Perform Fourier transform to get the frequency spectrum
    spectrum = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), 1 / RATE)
    
    # Get the magnitude of the spectrum
    magnitude = np.abs(spectrum)
    
    # Identify the fundamental frequency
    fundamental_idx = np.argmax(magnitude)
    fundamental_freq = frequencies[fundamental_idx]
    
    # Determine string and fret based on fundamental and harmonics
    for string_note, string_freq in guitar_strings.items():
        for fret in range(0, 24):
            expected_freq = string_freq * (2 ** (fret / 12))
            if abs(fundamental_freq - expected_freq) < 1:
                return string_note, fret
    return None, None

def print_tab(string, fret):
    strings_order = ['E4', 'B3', 'G3', 'D3', 'A2', 'E2']
    tab_lines = ["-" * 30 for _ in range(6)]
    if string in strings_order:
        string_index = strings_order.index(string)
        fret_pos = min(fret, 29)  # Ensures fret position is within the tab width
        tab_lines[string_index] = tab_lines[string_index][:fret_pos] + str(fret) + tab_lines[string_index][fret_pos+1:]
    
    # Print each string in reverse order (from highest pitch to lowest pitch)
    for i, line in enumerate(tab_lines):
        print(f"|{line}| {strings_order[i]}")

def process_audio_stream(stream):
    try:
        while True:
            data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
            frequency = librosa.yin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            frequency = np.mean(frequency)
            closest_note = closest_note_frequency(frequency)
            if closest_note:
                string, fret = determine_string_and_fret_harmonics(data)
                if string and fret is not None:
                    print(f"\nNote: {closest_note}, String: {string}, Fret: {fret}")
                    print_tab(string, fret)
    except KeyboardInterrupt:
        print("\nTerminating the program.")

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        process_audio_stream(stream)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
