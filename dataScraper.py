import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urlparse, urljoin
import re

# Configuration for Agricultural University Peshawar
BASE_URL = "https://www.aup.edu.pk"
ALLOWED_DOMAINS = ["www.aup.edu.pk", "aup.edu.pk"]
OUTPUT_DIR = "aup_data_txt"

def clean_text_content(text):
    """Cleans up text content."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.replace('\xa0', ' ')   # Replace non-breaking spaces
    text = text.replace('\n', ' ')     # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)   # Clean up any resulting multiple spaces
    return text.strip()

def remove_contact_info(text):
    """Removes phone numbers and office numbers from the text.

    This function uses regex patterns to remove sequences that match common
    phone number or office number formats.
    """
    # Pattern to match phone numbers in various formats:
    # e.g., +1 123-456-7890, (123) 456-7890, 1234567890, 123 456 7890, etc.
    phone_pattern = re.compile(
        r'(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?[\d\s\-]{7,}\d'
    )
    cleaned_text = re.sub(phone_pattern, '', text)

    # Remove lingering labels like "Phone:", "Tel:" or "Office:" if they appear without a number.
    label_pattern = re.compile(r'\b(?:Phone|Tel|Office):\s*', re.IGNORECASE)
    cleaned_text = re.sub(label_pattern, '', cleaned_text)

    return cleaned_text

def extract_page_content(soup, url):
    """Extracts nearly all text content from a page for RAG-based chatbot ingestion."""
    content = []

    # Add URL and page title as headers
    content.append(f"=== URL: {url} ===\n")
    title = soup.find('title')
    if title:
        content.append(f"=== Page Title ===\n{clean_text_content(title.text)}\n")

    # Extract text from the entire body.
    body = soup.find('body')
    if body:
        full_text = clean_text_content(body.get_text(separator=" "))
        # Remove phone/office numbers from the text.
        full_text = remove_contact_info(full_text)
        content.append(full_text)

    return "\n".join(content)

def is_valid_url(url):
    """Check if URL belongs to AUP domain. (Specific filtering removed.)"""
    try:
        parsed = urlparse(url)
        return parsed.netloc in ALLOWED_DOMAINS
    except Exception:
        return False

def save_content(content, filename):
    """Save the extracted content to a file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved content to: {filepath}")

def scrape_aup_website():
    """Main function to scrape the entire AUP website for data ingestion."""
    print(f"Starting to scrape {BASE_URL}")

    visited_urls = set()
    urls_to_visit = {BASE_URL}

    # Create directory for output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create a master file for all content
    master_file = os.path.join(OUTPUT_DIR, 'aup_website_data.txt')

    while urls_to_visit and len(visited_urls) < 200:  # Limit to 200 pages
        try:
            current_url = urls_to_visit.pop()
            if current_url in visited_urls:
                continue

            print(f"Processing: {current_url}")

            # Fetch page content
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract content and remove contact info
            content = extract_page_content(soup, current_url)

            # Save to master file
            with open(master_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(content)
                f.write(f"\n{'='*80}\n")

            # Find new links and add them if they belong to the allowed domain.
            for link in soup.find_all('a', href=True):
                url = urljoin(current_url, link['href'])
                if is_valid_url(url) and url not in visited_urls:
                    urls_to_visit.add(url)

            visited_urls.add(current_url)

            # Polite delay to avoid overloading the server
            time.sleep(2)

        except Exception as e:
            print(f"Error processing {current_url}: {str(e)}")
            continue

    print(f"\nScraping completed. Processed {len(visited_urls)} pages.")
    print(f"Content saved to: {master_file}")

if _name_ == "_main_":
    try:
        scrape_aup_website()
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")