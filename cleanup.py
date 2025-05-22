import os
import sys
import shutil

def fix_whisper_conflict():
    """Fix the conflict with whisper module"""
    # Get the site-packages directory
    site_packages_dir = None
    for path in sys.path:
        if "site-packages" in path:
            site_packages_dir = path
            break
    
    if not site_packages_dir:
        print("Couldn't find site-packages directory.")
        return False
    
    # Check for conflicting whisper.py file
    whisper_file = os.path.join(site_packages_dir, "whisper.py")
    if os.path.exists(whisper_file):
        print(f"Found conflicting whisper.py at {whisper_file}")
        
        # Backup the file
        backup_path = whisper_file + ".bak"
        try:
            shutil.move(whisper_file, backup_path)
            print(f"Moved conflicting whisper.py to {backup_path}")
            
            # Check for whisper package installation
            whisper_dir = os.path.join(site_packages_dir, "whisper")
            if not os.path.exists(whisper_dir) or not os.path.isdir(whisper_dir):
                print("OpenAI Whisper package not found. Installing...")
                return True  # Need to install OpenAI Whisper
            else:
                print("OpenAI Whisper package already installed.")
                return False  # No need to install
                
        except Exception as e:
            print(f"Error fixing whisper conflict: {str(e)}")
            return False
    else:
        # Check for whisper package installation
        whisper_dir = os.path.join(site_packages_dir, "whisper")
        if not os.path.exists(whisper_dir) or not os.path.isdir(whisper_dir):
            print("OpenAI Whisper package not found. Installing...")
            return True  # Need to install OpenAI Whisper
        
        print("No whisper.py conflict found.")
        return False

if __name__ == "__main__":
    need_install = fix_whisper_conflict()
    
    if need_install:
        print("\nPlease run the following command to install OpenAI Whisper:")
        print("pip install openai-whisper")