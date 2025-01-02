from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd
from datetime import datetime
from urllib.parse import quote

class IndeedJobScraper:
    def __init__(self):
        # Initialize Chrome driver
        self.driver = webdriver.Chrome()
        self.jobs_data = []
        
    def search_jobs(self, keyword, location="", num_pages=3):
        """Search for jobs with given keyword and location"""
        # Format the search URL
        base_url = "https://www.indeed.com/jobs"
        keyword_param = f"q={quote(keyword)}"
        location_param = f"l={quote(location)}" if location else ""
        
        # Construct the URL
        url = f"{base_url}?{keyword_param}&{location_param}"
        self.driver.get(url)
        time.sleep(3)
        
        for page in range(num_pages):
            try:
                # Scroll to load more jobs
                self.scroll_page()
                
                # Get all job cards
                job_cards = self.driver.find_elements(By.CLASS_NAME, "job_seen_beacon")
                
                for card in job_cards:
                    try:
                        # Extract job information directly from the card
                        job_info = self.extract_job_info(card)
                        if job_info:
                            self.jobs_data.append(job_info)
                            
                    except Exception as e:
                        print(f"Error extracting job info: {str(e)}")
                        continue
                
                # Try to click next page
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Next Page']")
                    if next_button.is_enabled():
                        next_button.click()
                        time.sleep(3)
                    else:
                        break
                except:
                    print("No more pages to load")
                    break
                    
            except Exception as e:
                print(f"Error on page {page + 1}: {str(e)}")
                break
                
    def extract_job_info(self, card):
        """Extract information from the job card"""
        try:
            title = card.find_element(By.CLASS_NAME, "jobTitle").text
            company = card.find_element(By.CLASS_NAME, "companyName").text
            location = card.find_element(By.CLASS_NAME, "companyLocation").text
            
            try:
                salary = card.find_element(By.CLASS_NAME, "salary-snippet").text
            except:
                salary = "Not available"
                
            try:
                description = card.find_element(By.CLASS_NAME, "job-snippet").text
            except:
                description = "Not available"
                
            return {
                'title': title,
                'company': company,
                'location': location,
                'salary': salary,
                'description': description,
                'scraped_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error extracting job details: {str(e)}")
            return None
            
    def scroll_page(self):
        """Scroll the page to load all job cards"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Calculate new scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Break if no more scrolling is possible
            if new_height == last_height:
                break
                
            last_height = new_height
            
    def save_to_csv(self, filename="indeed_jobs.csv"):
        """Save scraped jobs to CSV file"""
        df = pd.DataFrame(self.jobs_data)
        df.to_csv(filename, index=False)
        print(f"Saved {len(self.jobs_data)} jobs to {filename}")
        
    def close(self):
        """Close the browser"""
        self.driver.quit()

# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = IndeedJobScraper()
    
    try:
        # Search for jobs
        scraper.search_jobs(
            keyword="Software Engineer",  # Replace with your search keyword
            location="Remote",  # Replace with your location
            num_pages=3  # Number of pages to scrape
        )
        
        # Save results
        scraper.save_to_csv()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        # Close the browser
        scraper.close()
