"""
Generate the CareNote AI welcome audio using Microsoft Edge TTS.
Uses a deep, commanding Indian English male neural voice.
"""
import asyncio
import edge_tts

VOICE = "en-IN-PrabhatNeural"  # Deep Indian English male voice
TEXT = "Welcome to CareNote AI. . . Future AI in Healthcare."
OUTPUT = "frontend/welcome.mp3"

# SSML for fine-tuned dramatic delivery
SSML = """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-IN">
  <voice name="en-IN-PrabhatNeural">
    <prosody rate="-15%" pitch="-5%" volume="loud">
      Welcome to CareNote AI.
    </prosody>
    <break time="600ms"/>
    <prosody rate="-10%" pitch="-8%" volume="loud">
      Future AI in Healthcare.
    </prosody>
  </voice>
</speak>
""".strip()

async def main():
    communicate = edge_tts.Communicate(TEXT, VOICE, rate="-15%", pitch="-5Hz", volume="+20%")
    await communicate.save(OUTPUT)
    print(f"[OK] Welcome audio saved to {OUTPUT}")

asyncio.run(main())
