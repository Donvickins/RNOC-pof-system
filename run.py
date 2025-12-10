import subprocess as command
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: - %(message)s')
logger = logging.getLogger(__name__)

APP_DIRECTORY = "python_module"

try:
    command.run(
        ['python', 'App.py'],
        check=True,
        text=True,
        cwd=APP_DIRECTORY
    )
except command.CalledProcessError as e:
    logger.error(f'Failed to run program: Reason: {e}')
except KeyboardInterrupt:
    exit(1)
except FileNotFoundError:
    logger.error(f"Could not find the application directory '{APP_DIRECTORY}' or the script 'App.py' within it.")
    logger.info("Please ensure you are running this script from the project's root directory.")