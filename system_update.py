import os
import sys
import subprocess

# Function to install GitPython dynamically based on the system type
def install_gitpython():
    try:
        print("GitPython not found. Attempting to install GitPython...")
        if sys.platform.startswith("linux"):  # Check if the system is Linux-based
            try:
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call(["sudo", "apt-get", "install", "-y", "python3-git"])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install GitPython using apt: {e}")
                sys.exit(1)
        elif sys.platform == "win32":  # Windows-based system
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install GitPython using pip: {e}")
                sys.exit(1)
        else:
            print("Unsupported platform for automatic GitPython installation.")
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during GitPython installation: {e}")
        sys.exit(1)

# Check if gitpython is installed, and install it if not
try:
    import git
except ModuleNotFoundError:
    install_gitpython()
    import git  # Retry import after installation

# Detect OS and set repository path accordingly
if sys.platform.startswith("linux"):
    INSTALL_PATH = '/home/pi/02B-01'  # Path for Raspberry Pi
elif sys.platform == "win32":
    INSTALL_PATH = r'C:\Repos\02B-01'  # Path for Windows machine
else:
    print("Unsupported operating system.")
    sys.exit(1)

# GitHub repository URL
DATALOGGER_GITHUB = 'https://github.com/miniPCB/02B-01.git'

# Display program introduction
def print_intro():
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("miniPCB\nProgram: System Update")
    print(f"{DATALOGGER_GITHUB}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Display menu options
def print_menu():
    print("\n--- Select an Action ---")
    print("[1] Push System")
    print("[2] Pull System")
    print("[X] Exit")

# Function to stash local changes before pulling
def stash_local_changes(repo):
    if repo.is_dirty(untracked_files=True):
        print("Stashing local changes...")
        repo.git.stash()

# Function to pull the latest changes from the remote repository
def pull_from_github(repository, directory):
    try:
        # Check if the directory exists and is a Git repository
        if not os.path.exists(directory) or not os.path.exists(os.path.join(directory, '.git')):
            print(f"Cloning repository from {repository} to {directory}...")
            git.Repo.clone_from(repository, directory)
        else:
            # Pull the latest changes from the remote repository
            print("Pulling the latest changes from the remote repository...")
            repo = git.Repo(directory)

            # Stash local changes if there are any
            stash_local_changes(repo)

            # Specify rebase strategy to reconcile divergent branches
            origin = repo.remote(name='origin')
            repo.git.pull('--rebase')  # Rebase changes on top of the remote branch

        print("Pull completed successfully!")
    except git.exc.GitError as e:
        print(f"Git error occurred while pulling: {e}")
        if 'invalid path' in str(e):
            print("It looks like there is an invalid path in the repository for Windows (such as a file with a colon ':'). Please rename the file in the remote repository to avoid this issue.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

# Function to push local changes to GitHub
def push_to_github(repository, directory, commit_message):
    try:
        # Check if the directory exists and is a Git repository
        if not os.path.exists(directory) or not os.path.exists(os.path.join(directory, '.git')):
            print(f"Cloning repository from {repository} to {directory}...")
            git.Repo.clone_from(repository, directory)
        else:
            print("Repository exists.")

            repo = git.Repo(directory)

            # Debugging: Show the current branch
            print(f"Current branch: {repo.active_branch}")

            # Debugging: Show the remote URL
            origin = repo.remote(name='origin')
            print(f"Remote URL: {origin.url}")

            # Debugging: Check the repository status before committing
            print(repo.git.status())

            # Stage all changes
            print("Adding all local changes to staging area...")
            repo.git.add('--all')

            # Commit the changes before pulling
            print(f"Committing local changes with message: {commit_message}")
            repo.index.commit(commit_message)

            # Pull the latest changes from the remote repository
            print("Pulling the latest changes from the remote repository to ensure up-to-date...")
            repo.git.pull('--rebase')

        # Push the committed changes to GitHub
        print("Pushing committed changes to the remote repository...")
        origin.push()

        print("Push completed successfully!")
    except git.exc.GitError as e:
        print(f"Git error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

# Main program loop
def main():
    print_intro()
    while True:
        print_menu()
        choice = input("\nEnter menu selection: ").strip().lower()

        if choice == '1':
            commit_message = ""
            while not commit_message:  # Ensure the user provides a valid commit message
                commit_message = input("\nEnter commit message: ").strip()
                if not commit_message:
                    print("Commit message cannot be empty. Please try again.")
            push_to_github(DATALOGGER_GITHUB, INSTALL_PATH, commit_message)
            break

        elif choice == '2':
            print("\nPulling the latest changes from GitHub...")
            pull_from_github(DATALOGGER_GITHUB, INSTALL_PATH)
            break

        elif choice == 'x':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

# Run the program
if __name__ == "__main__":
    main()
