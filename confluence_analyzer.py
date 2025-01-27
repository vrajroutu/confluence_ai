import os
import logging
import requests
import yaml
from datetime import datetime
from azure.ai.vision import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from retrying import retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('confluence_analyzer.log'),
        logging.StreamHandler()
    ]
)

class ConfigManager:
    """Manage configuration from YAML file"""
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {str(e)}")
            raise

class ConfluenceArchitectureAnalyzer:
    def __init__(self, config):
        self.config = config
        self.vision_client = ImageAnalysisClient(
            self.config['azure']['vision_endpoint'],
            AzureKeyCredential(self.config['azure']['vision_key'])
        )
        self.openai_client = AzureOpenAI(
            api_key=self.config['azure']['openai_key'],
            azure_endpoint=self.config['azure']['openai_endpoint']
        )
        
    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def _confluence_api_call(self, url, params=None):
        """Generic Confluence API call with error handling"""
        try:
            response = requests.get(
                url,
                params=params,
                auth=(self.config['confluence']['user'], self.config['confluence']['api_token']),
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Confluence API error: {str(e)}")
            raise

    def search_confluence_space(self, software_name, space_key):
        """Search Confluence space for software-related content"""
        cql = f'space = "{space_key}" AND (text ~ "{software_name}" OR title ~ "{software_name}")'
        results = []
        start = 0
        
        while True:
            try:
                data = self._confluence_api_call(
                    f"{self.config['confluence']['base_url']}/content/search",
                    params={
                        "cql": cql,
                        "start": start,
                        "limit": 50,
                        "expand": "body.storage,children.attachment"
                    }
                )
                results.extend(data['results'])
                
                if data['size'] < 50:
                    break
                start += 50
            except Exception as e:
                logging.error(f"Search failed: {str(e)}")
                break
                
        return results

    def analyze_page_content(self, page_id):
        """Analyze a single Confluence page and its images"""
        try:
            page_data = self._confluence_api_call(
                f"{self.config['confluence']['base_url']}/content/{page_id}",
                params={"expand": "body.storage,children.attachment"}
            )
            
            text_content = page_data.get('body', {}).get('storage', {}).get('value', '')
            images = []
            
            # Process attachments
            for attachment in page_data.get('children', {}).get('attachment', {}).get('results', []):
                if attachment['mediaType'].startswith('image/'):
                    try:
                        analysis = self.vision_client.analyze(
                            image_url=attachment['_links']['download'],
                            features=["caption", "read", "denseCaptions"]
                        )
                        images.append({
                            'url': attachment['_links']['download'],
                            'description': analysis.caption.text,
                            'dense_captions': [dc.text for dc in analysis.dense_captions.list],
                            'ocr_text': ' '.join([line.text for block in analysis.read.blocks for line in block.lines]),
                            'confidence': analysis.caption.confidence
                        })
                    except Exception as e:
                        logging.error(f"Image analysis failed: {str(e)}")
                        continue
                        
            return {
                'title': page_data.get('title', 'Untitled'),
                'url': page_data.get('_links', {}).get('webui', ''),
                'text': text_content,
                'images': images
            }
        except Exception as e:
            logging.error(f"Page analysis failed: {str(e)}")
            return None

    def generate_architecture_report(self, software_name, pages, space_key):
        """Generate comprehensive architecture report using GPT-4"""
        system_prompt = f"""You are a senior software architect analyzing documentation for {software_name}. 
        Provide a detailed technical summary including:
        - System architecture and key components
        - Data flow and service interactions
        - Infrastructure and deployment patterns
        - Insights from architecture diagrams
        - Potential improvements and risks"""
        
        user_prompt = "Analyze the following documentation and images:\n\n"
        for idx, page in enumerate(pages):
            if not page:
                continue
            user_prompt += f"## Page {idx+1}: {page['title']}\n"
            user_prompt += f"Content: {page['text']}\n"
            if page['images']:
                user_prompt += "Diagram Insights:\n"
                for img in page['images']:
                    user_prompt += f"- {img['description']}\n"
                    user_prompt += f"  Detailed elements: {', '.join(img['dense_captions'])}\n"
                    user_prompt += f"  Extracted text: {img['ocr_text']}\n"
            user_prompt += "\n"
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['azure']['gpt_deployment'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=3000,
                top_p=0.95
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"GPT-4 analysis failed: {str(e)}")
            return None

    def generate_markdown_report(self, software_name, space_key, pages, summary):
        """Generate formatted Markdown report"""
        report = [
            f"# Software Architecture Report: {software_name}",
            f"**Confluence Space**: {space_key}  ",
            f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"**Analyzed Pages**: {len(pages)}\n"
        ]
        
        report.append("## Executive Summary\n")
        report.append(summary + "\n")
        
        report.append("## Detailed Page Analysis\n")
        for page in pages:
            if not page:
                continue
            report.append(f"### [{page['title']}]({page['url']})\n")
            report.append(f"**Key Content**:\n{page['text'][:1000]}...\n")
            if page['images']:
                report.append("#### Diagram Insights\n")
                for img in page['images']:
                    report.append(f"![Diagram]({img['url']})  ")
                    report.append(f"**Description**: {img['description']} (Confidence: {img['confidence']:.0%})  \n")
                    report.append(f"**Details**: {', '.join(img['dense_captions'])}  \n")
                    report.append(f"**Extracted Text**: {img['ocr_text']}\n")
        
        filename = f"{software_name.replace(' ', '_')}_architecture_report.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        return filename

    def analyze_software(self, software_name, space_key):
        """End-to-end analysis workflow"""
        try:
            logging.info(f"Starting analysis for {software_name} in space {space_key}")
            
            # Step 1: Search Confluence
            pages = self.search_confluence_space(software_name, space_key)
            if not pages:
                logging.warning("No relevant pages found")
                return None
                
            # Step 2: Analyze pages
            analyzed_pages = []
            for page in pages:
                result = self.analyze_page_content(page['id'])
                if result:
                    analyzed_pages.append(result)
            
            # Step 3: Generate summary
            summary = self.generate_architecture_report(software_name, analyzed_pages, space_key)
            if not summary:
                logging.error("Failed to generate summary")
                return None
                
            # Step 4: Create report
            report_file = self.generate_markdown_report(
                software_name, 
                space_key,
                analyzed_pages,
                summary
            )
            
            logging.info(f"Report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        config = ConfigManager().config
        analyzer = ConfluenceArchitectureAnalyzer(config)
        
        software_name = input("Enter software name to analyze: ").strip()
        space_key = input("Enter Confluence space key: ").strip()
        
        report_file = analyzer.analyze_software(software_name, space_key)
        if report_file:
            print(f"\nSuccess! Report saved to: {report_file}")
        else:
            print("\nAnalysis failed. Check logs for details.")
            
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        print("Analysis failed. Check logs for details.")