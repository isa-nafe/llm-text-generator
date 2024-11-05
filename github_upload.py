import os
import requests
import subprocess
from datetime import datetime

def verify_token_permissions(token):
    """Verify that the token has the necessary permissions."""
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Test the token by getting user information
    user_response = requests.get('https://api.github.com/user', headers=headers)
    if user_response.status_code != 200:
        print(f"Failed to authenticate with GitHub: {user_response.text}")
        return False
        
    # Check token's scopes
    scopes = user_response.headers.get('X-OAuth-Scopes', '').split(', ')
    required_scopes = ['repo']
    
    if not all(scope in scopes for scope in required_scopes):
        print(f"Token missing required scopes. Current scopes: {scopes}")
        print("Required scopes: repo")
        return False
        
    return True

def create_github_repo():
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("GitHub token not found in environment variables")
        return None, None
    
    if not verify_token_permissions(token):
        return None, None
        
    repo_name = "llm-text-generator"
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Get username
    user_response = requests.get('https://api.github.com/user', headers=headers)
    username = user_response.json()['login']
    
    # Create the repository
    repo_data = {
        'name': repo_name,
        'description': 'An interactive LLM-powered web application using Streamlit and Hugging Face Transformers',
        'private': False,
        'auto_init': False
    }
    
    create_response = requests.post('https://api.github.com/user/repos', 
                                  headers=headers, 
                                  json=repo_data)
    
    if create_response.status_code == 201:
        print(f"Repository {repo_name} created successfully!")
        return username, repo_name
    else:
        print(f"Failed to create repository: {create_response.text}")
        return None, None

def setup_and_push_to_github():
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("GitHub token not found in environment variables")
        return False
        
    username, repo_name = create_github_repo()
    
    if not username or not repo_name:
        return False
    
    # Use token in the remote URL for authentication
    remote_url = f'https://{token}@github.com/{username}/{repo_name}.git'
    
    commands = [
        ['git', 'init'],
        ['git', 'add', '.'],
        ['git', 'config', 'user.email', "replit@example.com"],
        ['git', 'config', 'user.name', "Replit User"],
        ['git', 'commit', '-m', f"Initial commit - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"],
        ['git', 'branch', '-M', 'main'],
        ['git', 'remote', 'add', 'origin', remote_url],
        ['git', 'push', '-u', 'origin', 'main']
    ]
    
    for cmd in commands:
        try:
            # Don't print the remote URL command as it contains the token
            if 'remote' not in cmd:
                print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if 'remote' not in cmd and result.stdout:
                print(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {str(e)}")
            return False
    return True

def main():
    if setup_and_push_to_github():
        print("Successfully uploaded project to GitHub")
        return True
    return False

if __name__ == "__main__":
    main()
