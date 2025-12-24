import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_existing_files(directory):
    """Returns a set of filenames currently in the directory."""
    return set(os.listdir(directory))


def get_new_file(before, after, extension=None):
    """
    Returns the new file in 'after' that's not in 'before'.
    If 'extension' is specified, filters new files by the given extension.
    """
    new_files = after - before
    if extension:
        new_files = {f for f in new_files if f.endswith(extension)}
    return new_files.pop() if new_files else None


def click_and_download_file(html_file: str, output_dir: str):
    """
    Opens an HTML file, clicks the download button, and saves the downloaded file to the output directory
    with the same name as the input file but with a `.json` extension.

    Args:
        html_file (str): Path to the HTML file.
        output_dir (str): Directory where the JSON file will be saved.
    """
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_argument('--headless=new')  # Use 'headless=new' for better compatibility
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

    # Set download preferences
    prefs = {
        "download.default_directory": os.path.abspath(output_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)

    # Enable downloads in headless mode using CDP
    try:
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {
                "behavior": "allow",
                "downloadPath": os.path.abspath(output_dir),
                "allowDownloadWithUserActivation": True
            }
        )
    except Exception as e:
        logger.error(f"Failed to set download behavior: {e}")
        driver.quit()
        return

    try:
        logger.info(f"Opening: {html_file}")
        driver.get(f"file://{os.path.abspath(html_file)}")

        # Wait for the download button to be clickable
        button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.v-btn--text"))
        )

        # Record existing files before download
        before_download = get_existing_files(output_dir)

        # Click the download button
        button.click()
        logger.info("Clicked the download button.")

        # Wait for the new .json file to appear in the output directory
        timeout = 60  # seconds
        wait_start = time.time()
        downloaded_file = None

        while time.time() - wait_start < timeout:
            after_download = get_existing_files(output_dir)
            new_file = get_new_file(before_download, after_download, extension='.json')
            if new_file:
                downloaded_file = new_file
                break
            time.sleep(1)

        if not downloaded_file:
            # If no .json file is found, check if any file exists after download initiation
            after_download = get_existing_files(output_dir)
            all_new_files = after_download - before_download
            if all_new_files:
                logger.warning(f"Unexpected new files detected: {all_new_files}")
            logger.error("Download timed out or .json file not found.")
            return

        downloaded_path = os.path.join(output_dir, downloaded_file)
        logger.info(f"Downloaded file detected: {downloaded_file}")

        # Ensure the file is fully written by checking its size remains constant
        initial_size = -1
        while True:
            current_size = os.path.getsize(downloaded_path)
            if current_size == initial_size:
                break
            initial_size = current_size
            time.sleep(0.5)

        # Rename the file to the expected name
        input_base_name = os.path.splitext(os.path.basename(html_file))[0]
        expected_filename = f"{input_base_name}.json"
        expected_path = os.path.join(output_dir, expected_filename)

        # Handle potential name conflicts
        if os.path.exists(expected_path):
            os.remove(expected_path)

        os.rename(downloaded_path, expected_path)
        logger.info(f"Download complete and renamed to: {expected_filename}")

    except Exception as e:
        logger.error(f"Error processing file {html_file}: {e}")

    finally:
        driver.quit()


def process_html_files(input_dir: str, output_dir: str):
    """
    Processes all HTML files in the input directory, clicking download buttons and saving the files.

    Args:
        input_dir (str): Directory containing HTML files.
        output_dir (str): Directory where JSON files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Iterate through all HTML files in the input directory
    html_files = [f for f in os.listdir(input_dir) if f.endswith('.html')]
    if not html_files:
        logger.warning(f"No HTML files found in the input directory: {input_dir}")
        return

    for html_file in html_files:
        html_path = os.path.join(input_dir, html_file)
        logger.info(f"Processing file: {html_path}")
        click_and_download_file(html_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Process all HTML files in a folder and download associated JSON files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder containing HTML files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the folder where JSON files will be saved."
    )
    args = parser.parse_args()

    process_html_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
