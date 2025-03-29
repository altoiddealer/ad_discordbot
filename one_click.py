import argparse
import hashlib
import os
import platform
import signal
import site
import subprocess
import sys

# Unchanging variables
script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)
home_install_path = os.path.join(script_dir, "installer_files")
home_conda_path = os.path.join(home_install_path, "conda")
home_conda_env_path = os.path.join(home_install_path, "env")
home_conda_bat = os.path.join(home_conda_path, "condabin", "conda.bat")

# Default environment
install_path = os.path.join(script_dir, "installer_files")
conda_path = os.path.join(install_path, "conda")
conda_env_path = os.path.join(install_path, "env")
env_flag = os.path.join(install_path, "user_env.txt")
project_url = "https://github.com/altoiddealer/ad_discordbot"
scripts_os = None # linux/macos/windows/wsl
parent_is_tgwui = False
is_tgwui_integrated = False

# Command-line flags
cmd_flags_path = os.path.join(script_dir, "CMD_FLAGS.txt")
if os.path.exists(cmd_flags_path):
    with open(cmd_flags_path, 'r') as f:
        CMD_FLAGS = ' '.join(line.strip().rstrip('\\').strip() for line in f if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'))
else:
    CMD_FLAGS = ''

flags = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update'])} {CMD_FLAGS}"


def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def is_linux():
    return sys.platform.startswith("linux")


def is_windows():
    return sys.platform.startswith("win")


def is_macos():
    return sys.platform.startswith("darwin")


def is_x86_64():
    return platform.machine() == "x86_64"


def is_installed():
    site_packages_path = None
    # look for discord
    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir and conda_env_path in sitedir:
            site_packages_path = sitedir
            break

    if site_packages_path:
        return os.path.isfile(os.path.join(site_packages_path, 'discord', '__init__.py'))
    else:
        return os.path.isdir(conda_env_path)


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd("conda", environment=True, capture_output=True).returncode == 0
    if not conda_exist:
        print("Conda is not installed. Exiting...")
        sys.exit(1)

    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit(1)


def get_current_commit():
    result = run_cmd("git rev-parse HEAD", capture_output=True, environment=True)
    return result.stdout.decode('utf-8').strip()


def is_fork_of(project_path, original_repo_url):
    try:
        upstream_result = run_cmd(f'git -C {project_path} config --get remote.origin.url', environment=True, capture_output=True)
        upstream_url = upstream_result.stdout.strip().decode() if upstream_result.stdout else ""

        # If upstream exists and matches the original repo URL, it's a fork
        if upstream_url and upstream_url.rstrip(".git") == original_repo_url.rstrip(".git"):
            return True

        return False
    except subprocess.CalledProcessError:
        return False  # If there's an error (e.g., not a git repo or no upstream), return False


def get_git_remote_url(project_path):
    try:
        result = run_cmd(f'git -C {project_path} config --get remote.origin.url', environment=True, capture_output=True)
        return result.stdout.strip()  # Remove any trailing whitespace
    except subprocess.CalledProcessError:
        return None  # Return None if the command fails (not a git repo, etc.)

def check_project():
    parent_is_tgwui = False
    is_tgwui_integrated = False

    bot_url = "https://github.com/altoiddealer/ad_discordbot"
    bot_git_url = "https://github.com/altoiddealer/ad_discordbot.git"
    tgwui_url = "https://github.com/oobabooga/text-generation-webui"
    tgwui_git_url = "https://github.com/oobabooga/text-generation-webui.git"
    supported_project_urls = [bot_url, bot_git_url, tgwui_url, tgwui_git_url]

    # Check if bot is running in a supported project
    project_path = os.path.dirname(install_path)
    project_url = get_git_remote_url(project_path)
    project_is_tgwui_fork = is_fork_of(project_path, tgwui_url)

    if (project_path == parent_dir) and (project_url not in supported_project_urls) and (not project_is_tgwui_fork):
        print_big_message(f"Bot is unexpectedly running in the environment of '{project_url}'.")
        print_big_message(f"Please refer to 'https://github.com/altoiddealer/ad_discordbot/wiki/installation'")
        print_big_message(f"Only attempt installing with 'text-generation-webui integration' if ad_discordbot is in it's directory.")
        sys.exit(1)

    # Check if bot is running as text-generation-webui integration.
    parent_url = get_git_remote_url(parent_dir)
    parent_is_tgwui_fork = is_fork_of(parent_dir, tgwui_git_url)
    if parent_is_tgwui_fork or parent_url in [tgwui_url, tgwui_git_url]:
        parent_is_tgwui = True
        if parent_url == project_url:
            is_tgwui_integrated = True
    
    return parent_is_tgwui, is_tgwui_integrated


def clear_cache():
    run_cmd("conda clean -a -y", environment=True)
    run_cmd("python -m pip cache purge", environment=True)


def print_big_message(message):
    message = message.strip()
    lines = message.split('\n')
    print("\n\n*******************************************************************")
    for line in lines:
        print("*", line)

    print("*******************************************************************\n\n")


def calculate_file_hash(file_path):
    p = os.path.join(script_dir, file_path)
    if os.path.isfile(p):
        with open(p, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    else:
        return ''


def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
    # Use the conda environment
    if environment:
        if is_windows():
            conda_bat_path = os.path.join(install_path, "conda", "condabin", "conda.bat")
            cmd = f'"{conda_bat_path}" activate "{conda_env_path}" >nul && {cmd}'
        else:
            conda_sh_path = os.path.join(install_path, "conda", "etc", "profile.d", "conda.sh")
            cmd = f'. "{conda_sh_path}" && conda activate "{conda_env_path}" && {cmd}'

    # Set executable to None for Windows, bash for everything else
    executable = None if is_windows() else 'bash'

    # Run shell commands
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env, executable=executable)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print(f"Command '{cmd}' failed with exit status code '{str(result.returncode)}'.\n\nExiting now.\nTry running the start/update script again.")
        sys.exit(1)

    return result


def generate_alphabetic_sequence(index):
    result = ''
    while index >= 0:
        index, remainder = divmod(index, 26)
        result = chr(ord('A') + remainder) + result
        index -= 1

    return result


def get_user_choice(question, options_dict):
    print()
    print(question)
    print()

    for key, value in options_dict.items():
        print(f"{key}) {value}")

    print()

    choice = input("Input> ").upper()
    while choice not in options_dict.keys():
        print("Invalid choice. Please try again.")
        choice = input("Input> ").upper()

    return choice


def restart_in_conda_env(env_path):
    """Restart the script in the specified Conda environment."""
    conda_python = os.path.join(env_path, "python.exe") if os.name == "nt" else os.path.join(env_path, "bin", "python")

    if not os.path.exists(conda_python):
        print(f"Error: Conda environment at {env_path} not found!")
        sys.exit(1)

    print(f"Restarting script in Conda environment: {env_path}")
    run_cmd(conda_python, environment=True)
    sys.exit()


def switch_to_launcher():
    launcher_name = f"start_{scripts_os}.bat"
    launcher_path = os.path.join(script_dir, launcher_name)
    print("Exiting one_click.py, and launching:", launcher_name)
    os.system(f'start "" "{launcher_path}"')  # Non-blocking execution
    sys.exit()


def install_bot():
    update_requirements(initial_installation=True, pull=False)


def update_requirements(initial_installation=False, pull=True):
    # Create .git directory if missing
    if not os.path.exists(os.path.join(script_dir, ".git")):
        run_cmd(
            "git init -b main && git remote add origin https://github.com/altoiddealer/ad_discordbot && "
            "git fetch && git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/main && "
            "git reset --hard origin/main && git branch --set-upstream-to=origin/main",
            environment=True,
            assert_success=True
        )

    requirements_file = "requirements.txt"
    if pull:
        print_big_message('Updating the local copy of the repository with "git pull"')

        # Hash files before pulling
        files_to_check = [
            'start_linux.sh', 'start_macos.sh', 'start_windows.bat', 'start_wsl.sh',
            'update_linux.sh', 'update_macos.sh', 'update_windows.bat', 'update_wsl.sh',
            'one_click.py'
        ]
        before_hashes = {file: calculate_file_hash(file) for file in files_to_check}

        # Perform the git pull
        run_cmd("git pull --autostash", assert_success=True, environment=True)

        # Check hashes after pulling
        after_hashes = {file: calculate_file_hash(file) for file in files_to_check}

        # Check for changes to installer files
        for file in files_to_check:
            if before_hashes[file] != after_hashes[file]:
                print_big_message(f"File '{file}' was updated during 'git pull'. Please run the script again.")

                sys.exit(1)

    print_big_message(f"Installing requirements from file: {requirements_file}")

    # Prepare the requirements file
    bot_requirements = open(requirements_file).read().splitlines()

    if not initial_installation:
        bot_requirements = [line for line in bot_requirements if '.whl' not in line]

    with open('temp_requirements.txt', 'w') as file:
        file.write('\n'.join(bot_requirements))

    # Workaround for git+ packages not updating properly.
    git_requirements = [req for req in bot_requirements if req.startswith("git+")]
    for req in git_requirements:
        url = req.replace("git+", "")
        package_name = url.split("/")[-1].split("@")[0].rstrip(".git")
        run_cmd(f"python -m pip uninstall -y {package_name}", environment=True)
        print(f"Uninstalled {package_name}")

    # Install/update the project requirements
    run_cmd("python -m pip install -r temp_requirements.txt --upgrade", assert_success=True, environment=True)

    # Clean up
    os.remove('temp_requirements.txt')
    clear_cache()


def launch_bot():
    run_cmd(f"python bot.py {flags}", environment=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conda-env-path", type=str, help="Path to the active Conda environment")
    args, unknown_args = parser.parse_known_args()
    # Check for any argument that starts with '--update-wizard-'
    for arg in unknown_args:
        if arg.startswith("--update-wizard-"):
            scripts_os = arg[len("--update-wizard-"):]
    # Write/update user_env.txt
    if args.conda_env_path:
        env_flag_path = os.path.join(script_dir, "installer_files", "user_env.txt")
        with open(env_flag_path, "w") as f:
            f.write(args.conda_env_path)
    # Update paths
    conda_env_path = os.path.abspath(args.conda_env_path)
    install_path = os.path.dirname(conda_env_path)
    # Check if bot is nested in TGWUI directory
    parent_is_tgwui, is_tgwui_integrated = check_project()
    # Add bot argument if bot is running as text-generation-webui integration.
    if is_tgwui_integrated:
        flags += " --is-tgwui-integrated"

    # Verifies we are in a conda environment
    check_env()

    if scripts_os: # OS name extracted from wizard flag
        print_big_message(f"Currently installed {'with text-generation-webui integration' if is_tgwui_integrated else 'as Standalone'}")
        options_dict = {}
        # Options based on current install status and environment
        if is_tgwui_integrated or parent_is_tgwui:
            # options_dict['S'] = 'Switch install method (Add/Remove TGWUI integration)'
            if is_tgwui_integrated:
                options_dict['S'] = 'Switch to standalone environment (remove TGWUI integration)'
            elif parent_is_tgwui:
                options_dict['S'] = 'Switch to text-generation-webui integration'
        # Always-present options
        options_dict.update({
            'U': 'Update the bot',
            'R': 'Revert local changes to repository files with \"git reset --hard\"',
            'N': 'Nothing (exit)'
        })

        while True:
            choice = get_user_choice("What would you like to do?", options_dict)

            if choice == 'S':
                # Switch to standalone environment (remove TGWUI integration)
                if is_tgwui_integrated:
                    print("Removing TGWUI integration")
                    new_conda_path = home_conda_path
                # Switch to TGWUI integration
                elif parent_is_tgwui:
                    print("Switching to text-generation-webui integration...")
                    new_conda_path = os.path.join(parent_dir, "installer_files", "conda")
                # Apply change and launch the os launcher
                with open(env_flag, "w") as f:
                    f.write(new_conda_path)
                switch_to_launcher()

            elif choice == 'U':
                update_requirements()
            elif choice == 'R':
                run_cmd(f'git -C "{script_dir}" reset --hard', assert_success=True, environment=True)
            elif choice == 'N':
                sys.exit()
    else:
        if not is_installed():
            install_bot()
            os.chdir(script_dir)

        if os.environ.get("LAUNCH_AFTER_INSTALL", "").lower() in ("no", "n", "false", "0", "f", "off"):
            print_big_message("Will now exit due to LAUNCH_AFTER_INSTALL.")
            sys.exit()

        # Launch the bot
        launch_bot()
