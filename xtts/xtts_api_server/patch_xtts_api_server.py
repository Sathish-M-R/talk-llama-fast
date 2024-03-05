import os, time, shutil, codecs

#  [file1, {needle1: replacement1, needle2: replacement2},     file2, {}]
def return_patch_files():
    file_paths_with_replacements = [
        ['server.py', 
            {'language = request.language[0:2]': """language = request.language[0:2]
            
            # CHECK: dont play if user is talking
            if STREAM_PLAY_SYNC:
                filename = 'xtts_play_allowed.txt'          # File name to be checked
                try:
                    if not os.path.isfile(filename):
                        print("File "+filename+" does not exist.")

                    else:
                        with open(filename, 'r') as fp:
                            data = fp.read().strip()          
                            play_allowed = int(data)                 

                        if play_allowed == 0: # CHECK: dont play if user is talking
                            print("user is speaking, xtts wont play")
                            return FileResponse(
                                path=output,
                                media_type='audio/wav',
                                filename="silence.wav",
                            )
                except Exception as e:
                    print(f"An error occurred: {e}")""",
    
            }
        ],
        
        ['RealtimeTTS/text_to_stream.py',
            {
                'import wave': """import wave
import os""",

                'def play_async(self,': """def check_for_stop(self, chunk):
        #Check if 'check_stop.txt' contains '1', stop streaming, and reset.
        filename = 'xtts_play_allowed.txt'          # File name to be checked
        value = None                         # Variable to store the content of the file (if any)
        try:
            if not os.path.isfile(filename):
                print("File "+filename+" does not exist.")

            else:
                with open(filename, 'r') as fp:
                    data = fp.read().strip()           # Read the entire file and remove leading/trailing whitespace characters
                    value = int(data)                  # Try converting the string to an integer

                if value == 0:                         # If the file contains '0'
                    if self.player:
                        self.player.mute(True)
                        self.player.stop(True)

                    with open(filename, 'w+') as fp:   # Reopen the file in writing mode ('w+')
                        fp.write('1\\n')                # Reset the file contents to '1' (allowed)
                        print("Stream stopped successfully!")
                        return 0
        except Exception as e:
            print(f"An error occurred: {e}")
                
            return 0
    
    def play_async(self,""",
    
                'on_audio_chunk = None,': 'on_audio_chunk = check_for_stop,',
                
                'on_sentence_synthesized(sentence)': 'on_sentence_synthesized(self, sentence)',
                
                'self.chunk_callback(chunk)': 'self.chunk_callback(self, chunk)'
            }
        ]
    ]
    
    return file_paths_with_replacements
    

def patch_files(file_paths_with_replacements):
    for file_path_with_replacements in file_paths_with_replacements:
        file_path = file_path_with_replacements[0]
        if os.path.exists(file_path):
            replacements = file_path_with_replacements[1]
            
            # Check if file has already been patched
            backup_file_path = file_path + '.bkp'
            if os.path.exists(backup_file_path):
                print(file_path+" was already patched before (.bkp file exists. If you want to patch it again - first restore the original file), skipping.")
                continue
            
            # Create backup copy of original file
            shutil.copy(file_path, backup_file_path)
            
            # Make replacements in file
            with codecs.open(file_path, encoding='utf-8', mode='r+') as f:
                file_contents = f.read()
                
                for needle, replacement in replacements.items():
                    file_contents = file_contents.replace(needle, replacement)
                    
                f.seek(0)
                f.write(file_contents)
                f.close()
                print(file_path+" successfully patched.")
        else:
            print(file_path+" is not found, skipping.")

    
if __name__ == "__main__":
    
    
    patch_files(return_patch_files())
    print ("\n\nSuccess. closing this window in 60 seconds.")
    time.sleep(60)