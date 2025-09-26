"""
Setup script for Clash Royale Bot

This script helps set up the development environment and install dependencies.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    directories = [
        "data/matches",
        "data/models", 
        "data/screenshots",
        "logs",
        "tests",
        "src/vision",
        "src/strategy",
        "src/automation",
        "src/learning",
        "src/recording"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ All directories created!")

def setup_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
data/matches/*.json
data/models/*.pth
logs/*.log
screenshots/*.png
screenshots/*.jpg

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✓ .gitignore created!")

def main():
    """Main setup function"""
    print("Setting up Clash Royale Bot...")
    
    # Create directories
    create_directories()
    
    # Setup gitignore
    setup_gitignore()
    
    # Install requirements
    if install_requirements():
        print("\n" + "="*50)
        print("Setup completed successfully!")
        print("="*50)
        print("\nNext steps:")
        print("1. Set up an Android emulator (MEmu recommended)")
        print("2. Install Clash Royale on the emulator")
        print("3. Run: python main.py")
        print("\nNote: Make sure the emulator window title matches 'Clash Royale'")
        print("or update the config.py file accordingly.")
    else:
        print("Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
