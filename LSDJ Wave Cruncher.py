import tkinter as tk
from tkinter import ttk, filedialog
import wave
import numpy as np
import aubio
import os
import pyaudio

# Constants for waveform and tone generation
MAX_VALUE = 15  # 4-bit maximum (0–F)
FRAME_SIZE = 32  # 32 samples per frame
TOTAL_FRAMES = 16  # 16 frames (512 samples total)
SAMPLE_RATE = 44100  # Audio sample rate
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BASE_FREQUENCY = 16.35  # Frequency for C0


def get_frequency(note, octave):
    """Calculate the frequency for a given note and octave based on C0."""
    note_index = NOTES.index(note)
    # Each semitone is 2^(1/12); each octave doubles the frequency.
    return BASE_FREQUENCY * (2 ** (note_index / 12)) * (2 ** octave)


class LSDJWaveConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("LSDJ Wave Converter v1.1")
        self.setup_variables()
        self.create_widgets()

    def setup_variables(self):
        """Initialize all Tkinter variables and internal variables."""
        self.normalize = tk.BooleanVar(value=True)
        self.channel = tk.IntVar(value=0)
        # Change default interpolation to "none"
        self.interpolation = tk.StringVar(value="none")
        self.analyze_only = tk.BooleanVar(value=False)
        # Waveform viewer variables
        self.current_frame = 0
        self.waveform = None  # 512-sample array after conversion
        # PyAudio instance for tone playback
        self.p = pyaudio.PyAudio()
        # Store full paths internally while showing shortened versions to the user
        self.full_input_path = ""
        self.full_output_path = ""

    def create_widgets(self):
        """Build the GUI layout."""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Input Settings")
        input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        input_frame.columnconfigure(2, weight=1)
        # Moved Browse button to the left
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(row=0, column=0)
        ttk.Label(input_frame, text="WAV File:").grid(row=0, column=1, sticky="w", padx=(5, 0))
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.grid(row=0, column=2, padx=5, sticky="ew")

        # Frequency Settings
        freq_frame = ttk.LabelFrame(main_frame, text="Frequency Detection")
        freq_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(freq_frame, text="Base Frequency:").grid(row=0, column=0, sticky="w")
        self.freq_entry = ttk.Entry(freq_frame, width=15)
        self.freq_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.freq_entry.insert(0, "auto")  # Set default to "auto"
        ttk.Button(freq_frame, text="Auto Detect", command=self.auto_detect).grid(row=0, column=2)

        # Processing Options
        opt_frame = ttk.LabelFrame(main_frame, text="Processing Options")
        opt_frame.grid(row=2, column=0, sticky="w", pady=5)
        ttk.Checkbutton(opt_frame, text="Normalize", variable=self.normalize).grid(row=0, column=0, sticky="w")
        ttk.Label(opt_frame, text="Channel:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(opt_frame, textvariable=self.channel, values=[0, 1], width=3).grid(row=1, column=1, sticky="w")
        ttk.Label(opt_frame, text="Interpolation:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(opt_frame, textvariable=self.interpolation,
                     values=["none", "linear", "exponential"], width=10).grid(row=2, column=1, sticky="w")

        # Output Section
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings")
        output_frame.grid(row=2, column=1, sticky="ew", pady=5)
        output_frame.columnconfigure(2, weight=1)
        # Moved Browse button to the left
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=0)
        ttk.Label(output_frame, text="Output File:").grid(row=0, column=1, sticky="w", padx=(5, 0))
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.grid(row=0, column=2, padx=5, sticky="ew")
        ttk.Checkbutton(output_frame, text="Analyze Only", variable=self.analyze_only).grid(row=1, column=0,
                                                                                            columnspan=3, sticky="w")
        # Tone Controls Section
        tone_frame = ttk.LabelFrame(main_frame, text="Tone Controls")
        tone_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(tone_frame, text="Note:").grid(row=0, column=0, padx=5)
        self.note_var = tk.StringVar(value="C")
        self.note_dropdown = ttk.Combobox(tone_frame, textvariable=self.note_var,
                                          values=NOTES, state="readonly", width=5)
        self.note_dropdown.grid(row=0, column=1, padx=5)
        ttk.Label(tone_frame, text="Octave:").grid(row=0, column=2, padx=5)
        self.octave_var = tk.IntVar(value=4)
        self.octave_dropdown = ttk.Combobox(tone_frame, textvariable=self.octave_var,
                                            values=list(range(8)), state="readonly", width=5)
        self.octave_dropdown.grid(row=0, column=3, padx=5)
        ttk.Button(tone_frame, text="Play Tone", command=self.play_tone_ui).grid(row=0, column=4, padx=5)
        ttk.Button(tone_frame, text="Play Frame", command=self.play_frame_ui).grid(row=0, column=5, padx=5)

        # Conversion Button
        ttk.Button(main_frame, text="Convert", command=self.process).grid(row=4, column=0, columnspan=2, pady=10)

        # Waveform Viewer Section
        viewer_frame = ttk.LabelFrame(main_frame, text="Waveform Viewer")
        viewer_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        self.top_container = tk.Canvas(viewer_frame, width=700, height=40, bg="black")
        self.top_container.pack()
        self.canvas = tk.Canvas(viewer_frame, width=700, height=190, bg="black")
        self.canvas.pack()
        self.bottom_container = tk.Canvas(viewer_frame, width=700, height=40, bg="black")
        self.bottom_container.pack()
        nav_frame = ttk.Frame(viewer_frame)
        nav_frame.pack(pady=5)
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_frame).grid(row=0, column=0)
        ttk.Button(nav_frame, text="Animate Frames", command=self.animate_frames).grid(row=0, column=1)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_frame).grid(row=0, column=2)
        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0F")
        self.frame_label.grid(row=1, column=0, columnspan=3)

        # Status Bar
        self.status = ttk.Label(main_frame, text="Ready", foreground="gray")
        self.status.grid(row=6, column=0, columnspan=2)

    def browse_input(self):
        """Open file dialog to select input WAV file."""
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            # Store full path internally while showing only the filename
            self.full_input_path = path
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, os.path.basename(path))
            base = os.path.splitext(path)[0]
            # Automatically set output file based on input file
            self.full_output_path = base + ".snt"
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(self.full_output_path)))

    def browse_output(self):
        """Open file dialog to select output SNT file and display parent folder and file name."""
        path = filedialog.asksaveasfilename(
            defaultextension=".snt",
            filetypes=[("SNT files", "*.snt"), ("All files", "*.*")]
        )
        if path:
            self.full_output_path = path
            parent = os.path.basename(os.path.dirname(path))
            filename = os.path.basename(path)
            display_text = os.path.join(parent, filename)
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, display_text)

    def auto_detect(self):
        """Set frequency entry to 'auto' for auto-detection."""
        self.freq_entry.delete(0, tk.END)
        self.freq_entry.insert(0, "auto")

    def process(self):
        """Handle the conversion process."""
        try:
            # Use the stored full input and output paths
            input_path = self.full_input_path
            output_path = self.full_output_path
            if not input_path or not os.path.exists(input_path):
                raise ValueError("Please select a valid input file")
            # Read and process audio data
            rate, data = self.read_wav(input_path)
            data = self.select_channel(data)
            # Frequency detection
            freq = self.get_frequency(data, rate)
            self.freq_entry.delete(0, tk.END)
            self.freq_entry.insert(0, f"{freq:.2f}")
            # Resample and process
            samples = self.resample_data(data, freq, rate)
            if self.normalize.get():
                samples = self.normalize_samples(samples)
            bitcrushed = self.bitcrush(samples)
            if len(bitcrushed) != 512:
                raise ValueError("Bitcrushed data must have 512 samples")
            # Store waveform for viewing and playback (512 samples: 16 frames × 32 samples)
            self.waveform = bitcrushed.tolist()
            self.current_frame = 0
            self.update_display()
            if not self.analyze_only.get():
                self.save_snt(bitcrushed, output_path)
            self.update_status("Conversion successful!", "green")
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")

    def read_wav(self, path):
        """Read and normalize WAV file data (supports 8, 16, 24, and 32-bit files)."""
        with wave.open(path, 'rb') as wav_file:
            rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            samp_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            if samp_width not in [1, 2, 3, 4]:
                raise ValueError("Only 8, 16, 24, or 32-bit WAV files supported")
            raw_data = wav_file.readframes(n_frames)
        if samp_width == 1:
            dtype = np.int8
            max_val = np.iinfo(np.int8).max
            data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
        elif samp_width == 2:
            dtype = np.int16
            max_val = np.iinfo(np.int16).max
            data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
        elif samp_width == 3:
            a = np.frombuffer(raw_data, dtype=np.uint8)
            a = a.reshape(-1, 3)
            data = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8) | (a[:, 2].astype(np.int32) << 16))
            mask = 1 << 23
            data = np.where(data & mask, data - (1 << 24), data).astype(np.float32)
            max_val = (1 << 23) - 1
        elif samp_width == 4:
            dtype = np.int32
            max_val = np.iinfo(np.int32).max
            data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
        data /= max_val  # Normalize to [-1.0, 1.0]
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        return rate, data

    def select_channel(self, data):
        """Select audio channel for stereo files."""
        if data.ndim == 1:
            return data
        channel = self.channel.get()
        if channel >= data.shape[1]:
            raise ValueError(f"Channel {channel} not available")
        return data[:, channel]

    def get_frequency(self, data, sample_rate):
        """Handle frequency input/auto-detection."""
        input_val = self.freq_entry.get().strip().lower()
        if input_val == "auto":
            return self.detect_pitch(data, sample_rate)
        try:
            return float(input_val)
        except ValueError:
            return self.note_to_freq(input_val)

    def detect_pitch(self, data, sample_rate):
        """Accurate pitch detection using aubio."""
        frame_size = 512
        hop_size = 256
        if len(data) < hop_size:
            raise ValueError("Audio file too short for analysis (minimum 256 samples required)")
        pitch_detector = aubio.pitch("yin", frame_size, hop_size, sample_rate)
        pitch_detector.set_unit("Hz")
        pitch_detector.set_tolerance(0.8)
        pitches = []
        for i in range(0, len(data) - hop_size + 1, hop_size):
            chunk = data[i:i + hop_size]
            pitch = pitch_detector(chunk)[0]
            if 20 < pitch < 2000:
                pitches.append(pitch)
        if not pitches:
            raise ValueError("No detectable pitch found in audio")
        return np.median(pitches)

    def resample_data(self, data, freq, sample_rate):
        """
        Resample audio to LSDJ's 512-sample format using the selected interpolation option.
        """
        target_samples = 512
        original_indices = np.arange(len(data))
        interp_option = self.interpolation.get().lower()

        if interp_option == "linear":
            new_indices = np.linspace(0, len(data) - 1, target_samples)
            return np.interp(new_indices, original_indices, data)
        elif interp_option == "exponential":
            new_indices = np.logspace(0, np.log10(len(data) - 1), target_samples)
            return np.interp(new_indices, original_indices, data)
        else:
            indices = np.arange(target_samples)
            indices = np.clip(indices, 0, len(data) - 1)
            return data[indices]

    def normalize_samples(self, samples):
        """Normalize audio to maximum amplitude."""
        peak = np.max(np.abs(samples))
        return samples / peak if peak > 0 else samples

    def bitcrush(self, samples):
        """Convert samples to 4-bit LSDJ format."""
        scaled = (samples + 1.0) * 8  # Convert [-1, 1] to [0, 16]
        crushed = np.clip(scaled, 0, 15).astype(int)
        return np.where(crushed > 14, 15, crushed)

    def save_snt(self, data, path):
        """Save as packed 4-bit SNT format."""
        if len(data) != 512:
            raise ValueError("Invalid sample count for SNT format")
        with open(path, "wb") as f:
            for i in range(0, 512, 2):
                byte = (data[i] << 4) | data[i + 1]
                f.write(bytes([byte]))

    def note_to_freq(self, note):
        """Convert musical note to frequency (A4 = 440Hz)."""
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        try:
            octave = int(note[-1])
            name = note[:-1].upper().replace("S", "#")
            semitones = notes.index(name) + (octave - 4) * 12
            return 440.0 * (2 ** (semitones / 12))
        except (ValueError, IndexError):
            raise ValueError(f"Invalid note format: {note}")

    def update_status(self, message, color="gray"):
        """Update the status bar with a message."""
        self.status.config(text=message, foreground=color)
        self.root.after(5000, lambda: self.status.config(text="Ready", foreground="gray"))

    # --- Waveform Viewer Functions ---
    def draw_waveform(self):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x_scale = 22
        label_spacing = 10
        for i in range(33):
            x = i * x_scale
            self.canvas.create_line(x, 0, x, canvas_height, fill="gray", dash=(2, 2))
        for i in range(MAX_VALUE + 1):
            y = 20 + i * label_spacing
            self.canvas.create_line(0, y, canvas_width, y, fill="gray", dash=(2, 2))
        if not self.waveform:
            return
        start = self.current_frame * FRAME_SIZE
        for i in range(FRAME_SIZE):
            x = i * x_scale
            value = self.waveform[start + i]
            y = 20 + (MAX_VALUE - value) * label_spacing
            self.canvas.create_rectangle(x, y, x + x_scale, y + label_spacing, fill="blue")

    def display_hex_values(self):
        self.top_container.delete("all")
        self.bottom_container.delete("all")
        if not self.waveform:
            return
        start = self.current_frame * FRAME_SIZE
        x_scale = 22
        for i in range(16):
            x = i * x_scale * 2
            self.top_container.create_text(x + x_scale, 10, text=f"{self.waveform[start + i]:X}", fill="white")
        for i in range(16, FRAME_SIZE):
            x = (i % 16) * x_scale * 2
            self.bottom_container.create_text(x + x_scale, 10, text=f"{self.waveform[start + i]:X}", fill="white")

    def update_display(self):
        self.draw_waveform()
        self.display_hex_values()
        self.frame_label.config(text=f"Frame: {self.current_frame:X}/0F")

    def next_frame(self):
        if not self.waveform:
            return
        if self.current_frame < TOTAL_FRAMES - 1:
            self.current_frame += 1
            self.update_display()

    def prev_frame(self):
        if not self.waveform:
            return
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()

    def animate_frames(self):
        def next_frame_animation(frame):
            if not self.waveform:
                return
            if frame < TOTAL_FRAMES:
                self.current_frame = frame
                self.update_display()
                self.root.after(250, next_frame_animation, frame + 1)
        next_frame_animation(0)

    # --- Tone Playback Functions ---
    def play_frame(self, frequency=None):
        """
        Play the current frame as a periodic waveform at the given frequency.
        If frequency is not provided, it defaults to SAMPLE_RATE/FRAME_SIZE.
        """
        if frequency is None:
            frequency = SAMPLE_RATE / FRAME_SIZE
        period_samples = int(SAMPLE_RATE / frequency)
        start = self.current_frame * FRAME_SIZE
        frame_data = np.array(self.waveform[start:start + FRAME_SIZE], dtype=np.float32)
        orig_indices = np.linspace(0, FRAME_SIZE, num=FRAME_SIZE, endpoint=False)
        new_indices = np.linspace(0, FRAME_SIZE, num=period_samples, endpoint=False)
        resampled_cycle = np.interp(new_indices, orig_indices, frame_data)
        num_cycles = int(np.ceil(SAMPLE_RATE / period_samples))
        full_wave = np.tile(resampled_cycle, num_cycles)[:SAMPLE_RATE]
        normalized = (full_wave / MAX_VALUE) * 2 - 1
        audio_data = (normalized * 32767).astype(np.int16).tobytes()
        stream = self.p.open(format=pyaudio.paInt16, channels=1,
                             rate=SAMPLE_RATE, output=True)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()

    def play_tone(self, frequency):
        """
        Play a tone generated by modulating the entire waveform with a sine wave
        at the given frequency.
        """
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        total_samples = len(self.waveform)
        audio_waveform = np.array([
            self.waveform[int(i * total_samples / len(t))]
            for i in range(len(t))
        ])
        audio_waveform = (audio_waveform / MAX_VALUE) * 2 - 1
        audio_waveform = np.sin(2 * np.pi * frequency * t) * audio_waveform
        audio_data = (audio_waveform * 32767).astype(np.int16).tobytes()
        stream = self.p.open(format=pyaudio.paInt16, channels=1,
                             rate=SAMPLE_RATE, output=True)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()

    def play_tone_ui(self):
        """Compute frequency from tone controls and play the full tone."""
        note = self.note_var.get()
        octave = self.octave_var.get()
        frequency = get_frequency(note, octave)
        self.play_tone(frequency)

    def play_frame_ui(self):
        """Compute frequency from tone controls and play only the current frame."""
        note = self.note_var.get()
        octave = self.octave_var.get()
        frequency = get_frequency(note, octave)
        self.play_frame(frequency)


if __name__ == "__main__":
    root = tk.Tk()
    app = LSDJWaveConverter(root)
    root.mainloop()
