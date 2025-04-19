import pyaudio, soundfile
import numpy as np

SAMPLERATE = 48000

def pan_frames(frames, factor):
    panned = np.copy(frames)
    if factor > 0:
        panned[:, 1] += frames[:, 0] * factor
        panned[:, 0] *= 1-factor
    elif factor < 0:
        factor *= -1
        panned[:, 0] += frames[:, 1] * factor
        panned[:, 1] *= 1-factor
    return panned

class SoundData:
    def __init__(self, filename):
        self.data, self.samplerate = soundfile.read(filename, dtype = "float32", always_2d = True)
        self.length = len(self.data)
    
    def get_frames(self, startIndex, n, startVolume=1.0, endVolume=1.0, startPitch=1.0, endPitch=1.0, pan=0.0, loop=False):
        startPitch *= self.samplerate / SAMPLERATE
        endPitch *= self.samplerate / SAMPLERATE
        pitch = np.linspace(startPitch, endPitch, n)

        indices = np.zeros(n)
        index = 0
        for i in range(n):
            indices[i] = startIndex + index
            if loop: indices[i] %= self.length
            index += pitch[i]
        indices = indices[indices < self.length - 1]

        floorIndices = np.floor(indices).astype("int32")
        ceilIndices = np.ceil(indices).astype("int32")
        interpolation = self.mono_to_stereo(indices - floorIndices)
        
        floorData = self.data[floorIndices]
        ceilData = self.data[ceilIndices]
        frames = (ceilData - floorData) * interpolation + floorData

        startVolume = max(0, startVolume)
        endVolume = max(0, endVolume)
        frames *= self.mono_to_stereo(np.linspace(startVolume, endVolume, len(frames)))

        frames = pan_frames(frames, pan)

        while len(frames) < n:
            frames = np.append(frames, [[0, 0]], axis = 0)
        return frames, startIndex + index
    
    def mono_to_stereo(self, array):
        return np.repeat(array[:, np.newaxis], 2, axis = 1)

class SoundLibrary:
    def __init__(self):
        self.sounds = {}
    
    def generate_sound(self, filename, volume=1.0, pitch=1.0, pan=0.0, id=None, loop=False):
        if id is None: id = filename
        return SoundInstance(self.get_sound_data(filename), id, volume, pitch, pan, loop)
    
    def get_sound_data(self, filename):
        if not filename in self.sounds:
            self.sounds[filename] = SoundData(filename)
        return self.sounds[filename]

class SoundInstance:
    def __init__(self, data, id, volume=1.0, pitch=1.0, pan=0.0, loop=False):
        self.id = id
        
        self.volume = volume
        self.previousVolume = volume
        self.pan = pan
        self.pitch = pitch
        self.previousPitch = pitch

        self.data = data
        self.index = 0
        self.loop = loop

        self.toRemove = False
    
    def get_frames(self, n):
        frames, self.index = self.data.get_frames(self.index, n, self.previousVolume, self.volume, self.previousPitch, self.pitch, self.pan, self.loop)
        if self.loop: self.index %= self.data.length
        self.previousVolume = self.volume
        self.previousPitch = self.pitch
        return frames
    
    def finished(self):
        return self.toRemove or (not self.loop) and self.index > self.data.length
    
    def remove(self):
        self.volume = 0
        self.toRemove = True
    
    def get_time(self):
        return self.index / self.data.samplerate

class SoundPlayer:
    def __init__(self):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.open_stream()
        self.soundLibrary = SoundLibrary()
        self.currentSounds = []
    
    def open_stream(self):
        try:
            self.stream = self.pyaudio.open(
                format = pyaudio.paFloat32,
                channels = 2,
                rate = SAMPLERATE,
                output = True,
                stream_callback = self.callback
            )
        except OSError:
            return

    def quit(self):
        if self.stream != None: self.stream.close()
        self.pyaudio.terminate()
    
    def mix(self, data, toMix):
        data += np.clip(toMix, -1.0 - data, 1.0 - data)
    
    def callback(self, inData, frameCount, timeInfo, status):
        data = np.zeros((frameCount, 2), dtype = "float32")
        for sound in list(self.currentSounds):
            self.mix(data, sound.get_frames(frameCount))

        self.check_sounds()
        return data, pyaudio.paContinue
    
    def play_sound(self, filename, volume=1.0, pitch=1.0, pan=0.0, id=None, loop=False):
        self.currentSounds.append(self.soundLibrary.generate_sound(filename, volume, pitch, pan, id, loop))
    
    def get_sounds(self, id):
        sounds = []
        for sound in list(self.currentSounds):
            if sound.id == id: sounds.append(sound)
        return sounds
    
    def get_sound(self, id):
        return self.get_sounds(id)[0]
    
    def stop_sounds(self, id):
        for sound in self.get_sounds(id): sound.remove()
    
    def check_sounds(self):
        for sound in list(self.currentSounds):
            if sound.finished(): self.currentSounds.remove(sound)

class MusicPlayer:
    def __init__(self, soundPlayer):
        self.soundPlayer = soundPlayer

        self.id = "music"
        self.bpm = 120
    
    def play(self, filename, bpm, volume=1.0, pitch=1.0, pan=0.0):
        self.bpm = bpm

        self.soundPlayer.stop_sounds(self.id)
        self.soundPlayer.play_sound(
            filename,
            volume,
            pitch,
            pan,
            self.id,
            True
        )
    
    def get_time(self):
        return (
            self.soundPlayer.get_sound(self.id).get_time()
            * (self.bpm / 60)
        )
