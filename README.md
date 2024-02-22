talk-llama-fast

Early pre beta!

based on talk-llama https://github.com/ggerganov/whisper.cpp

I added:
- xTTSv2 support;
- UTF8 and Russian;
- Speed-ups: streaming for generation, streaming for xtts, aggresive VAD;
- commands: Google, stop, regenerate, reset.
- generation interription when user is speaking.

i used: 
- whsiper-cpp ggml-medium-q5_0.bin
- mistral-7b-instruct-v0.2.Q6_K.gguf
- xTTSv2 server streaming-mode

- It's just a proof of work.
- Documentation is still TODO.
- Android version is TODO.

All running on 3060 12 GB vram, but i guess you can try with 8 GB.


I uploaded modified cpp files, but they still have some hardcoded paths (TODO, c:\\DATA\\LLM\\xtts\\xtts_play_allowed.txt), so i guess they won't work if you don't modify them manually

I uploaded win64 exe and bat files in realeses. You can try, but it still have some hardcoded paths (TODO)

Google search server is based on langchain google-serp, i will upload source code later.


xtts better coma handling:

c:\Users\[username]\miniconda3\Lib\site-packages\stream2sentence\stream2sentence.py

line 191, replace 

'''
sentence_delimiters = '.?!;:,\n…)]}。'
with
sentence_delimiters = '.?!;:\n…)]}。'
sentences with real coma sound nicer
'''