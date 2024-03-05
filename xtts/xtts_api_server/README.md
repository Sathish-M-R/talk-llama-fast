# Patcher for xtts to support interruptions (stop on user speech)

## Patch
- Stop xtts_api_server if running.
- Copy 2 files (patch_xtts_api_server.py, restore_xtts_api_server.py) to /xtts_api_server directory. Usually it is somewhere here:
c:\Users\[USER]\miniconda3\Lib\site-packages\xtts_api_server
- Double click patch_xtts_api_server.py to patch files (or run in cmd: python patch_xtts_api_server.py). Check that there are no errors. It will create .bkp files, then will patch original .py files.

## Restore
If you need to restore original files run:
restore_xtts_api_server.py

If you need to update xtts_api_server or need to re-patch it (patcher won't patch already patched files):
- run restore_xtts_api_server.py
- update xtts_api_server using pip or git
- run patch_xtts_api_server.py


## Modified files:
- server.py
- \RealtimeTTS\text_to_stream.py

Full list of modificitaions can be seen in patch_xtts_api_server, it has simple needle->replacement method.