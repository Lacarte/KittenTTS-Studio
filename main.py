from kittentts import KittenTTS
import soundfile as sf
import time

m = KittenTTS("KittenML/kitten-tts-mini-0.8")
#m = KittenTTS("KittenML/kitten-tts-micro-0.8")
#m = KittenTTS("KittenML/kitten-tts-nano-0.8-fp32")

prompt = """
The reports of my death are greatly exaggerated, but the reports of AI taking all our jobs... are somehow even more so.
"""

# available_voices : ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']

voice = 'Jasper'

# Token approximation (words * 1.3 is a rough heuristic)
words = len(prompt.split())
approx_tokens = int(words * 1.3)

print(f"Prompt      : {prompt.strip()}")
print(f"Word count  : {words}")
print(f"~Tokens     : {approx_tokens}")
print(f"Voice       : {voice}")
print(f"Generating...")

start = time.perf_counter()
audio = m.generate(prompt, voice=voice)
end = time.perf_counter()

duration_generated = len(audio) / 24000  # audio length in seconds
inference_time = end - start
rtf = inference_time / duration_generated  # real-time factor

print(f"\n--- Timing ---")
print(f"Inference time : {inference_time:.3f}s")

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
