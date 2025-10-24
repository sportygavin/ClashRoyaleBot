#!/usr/bin/env python3
"""
Simple deployment script for Clash Royale Bot
Creates a new version tag and pushes to GitHub without changing any code.
"""

import subprocess
import sys
import datetime
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        sys.exit(1)

def get_next_version():
    """Get the next version number based on existing tags."""
    try:
        # Get all version tags
        result = subprocess.run(
            "git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1",
            shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Parse the latest version
            latest_tag = result.stdout.strip()
            version_parts = latest_tag[1:].split('.')  # Remove 'v' prefix
            major, minor, patch = map(int, version_parts)
            
            # Increment patch version
            new_patch = patch + 1
            return f"v{major}.{minor}.{new_patch}"
        else:
            # No existing version tags, start with v1.0.0
            return "v1.0.0"
            
    except Exception as e:
        print(f"⚠️  Could not determine next version, using v1.0.0: {e}")
        return "v1.0.0"

def main():
    """Main deployment function."""
    print("🚀 Clash Royale Bot Deployment Script")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ Not in a git repository. Please run this script from the project root.")
        sys.exit(1)
    
    # Check if there are any uncommitted changes
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("⚠️  Warning: There are uncommitted changes in the working directory.")
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Deployment cancelled.")
            sys.exit(0)
    
    # Get current branch
    result = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
    current_branch = result.stdout.strip()
    print(f"📍 Current branch: {current_branch}")
    
    # Get next version
    version = get_next_version()
    print(f"📦 Next version: {version}")
    
    # Confirm deployment
    print(f"\n🎯 Ready to deploy version {version}")
    print("This will:")
    print(f"  - Create a new tag: {version}")
    print(f"  - Push the tag to GitHub")
    print(f"  - Push any local commits to origin/{current_branch}")
    
    response = input("\nProceed with deployment? (y/N): ")
    if response.lower() != 'y':
        print("Deployment cancelled.")
        sys.exit(0)
    
    # Create and push the tag
    run_command(f"git tag -a {version} -m 'Release {version} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'", 
                f"Creating tag {version}")
    
    # Push commits to origin
    run_command(f"git push origin {current_branch}", 
                f"Pushing commits to origin/{current_branch}")
    
    # Push tags to origin
    run_command("git push origin --tags", 
                "Pushing tags to GitHub")
    
    print("\n🎉 Deployment completed successfully!")
    print(f"📋 Version {version} has been deployed to GitHub")
    print(f"🔗 You can view the release at: https://github.com/YOUR_USERNAME/ClashRoyaleBot/releases/tag/{version}")
    print("\nNote: Replace 'YOUR_USERNAME' with your actual GitHub username in the URL above.")

if __name__ == "__main__":
    main()
