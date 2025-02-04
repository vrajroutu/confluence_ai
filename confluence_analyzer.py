import os
import logging
import requests
import yaml
from datetime import datetime
from azure.ai.vision import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from retrying import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("confluence_analyzer.log"),
        logging.StreamHandler()
    ]
)


class ConfigManager:
    """Manage configuration loaded from a YAML file."""
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_and_validate_config()

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load YAML configuration and validate required fields."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config file: {e}", exc_info=True)
            raise

        # Basic validation for required keys (extend as needed)
        required_fields = {
            "azure": ["vision_endpoint", "vision_key", "openai_key", "openai_endpoint", "gpt_deployment"],
            "confluence": ["base_url", "user", "api_token"]
        }
        for section, fields in required_fields.items():
            if section not in config:
                raise ValueError(f"Missing '{section}' section in configuration.")
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing field '{field}' in configuration section '{section}'.")
        return config


class ConfluenceArchitectureAnalyzer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        # Initialize Azure Vision client
        self.vision_client = ImageAnalysisClient(
            self.config["azure"]["vision_endpoint"],
            AzureKeyCredential(self.config["azure"]["vision_key"])
        )
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=self.config["azure"]["openai_key"],
            azure_endpoint=self.config["azure"]["openai_endpoint"]
        )
        # Create a persistent session for Confluence API calls
        self.session = requests.Session()
        self.session.auth = (self.config["confluence"]["user"], self.config["confluence"]["api_token"])
        self.session.headers.update({"Content-Type": "application/json"})

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def _confluence_api_call(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic Confluence API call with retry and error handling."""
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Confluence API error for URL {url} with params {params}: {e}", exc_info=True)
            raise

    def search_confluence_space(self, software_name: str, space_key: str) -> List[Dict[str, Any]]:
        """
        Search Confluence space for content related to a software name.
        Returns a list of page results.
        """
        cql = f'space = "{space_key}" AND (text ~ "{software_name}" OR title ~ "{software_name}")'
        results: List[Dict[str, Any]] = []
        start = 0
        limit = 50
        base_url = self.config["confluence"]["base_url"]
        search_url = f"{base_url}/content/search"

        while True:
            try:
                data = self._confluence_api_call(
                    search_url,
                    params={
                        "cql": cql,
                        "start": start,
                        "limit": limit,
                        "expand": "body.storage,children.attachment"
                    }
                )
                batch = data.get("results", [])
                results.extend(batch)

                # If fewer results than the limit were returned, we've reached the end
                if len(batch) < limit:
                    break
                start += limit
            except Exception as e:
                logging.error(f"Search failed at start {start}: {e}", exc_info=True)
                break

        logging.info(f"Found {len(results)} pages matching search criteria.")
        return results

    def analyze_page_content(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single Confluence page including text and image attachments.
        Returns a dictionary with page analysis or None if an error occurs.
        """
        try:
            base_url = self.config["confluence"]["base_url"]
            page_url = f"{base_url}/content/{page_id}"
            page_data = self._confluence_api_call(
                page_url,
                params={"expand": "body.storage,children.attachment"}
            )

            text_content = page_data.get("body", {}).get("storage", {}).get("value", "")
            images: List[Dict[str, Any]] = []

            attachments = page_data.get("children", {}).get("attachment", {}).get("results", [])
            for attachment in attachments:
                media_type = attachment.get("mediaType", "")
                if media_type.startswith("image/"):
                    image_url = attachment.get("_links", {}).get("download", "")
                    if not image_url:
                        continue
                    try:
                        # Analyze image using Azure Vision
                        analysis = self.vision_client.analyze(
                            image_url=image_url,
                            features=["caption", "read", "denseCaptions"]
                        )
                        caption = analysis.caption.text if analysis.caption else "No caption"
                        dense_captions = [dc.text for dc in analysis.dense_captions.list] if analysis.dense_captions and analysis.dense_captions.list else []
                        ocr_text = " ".join([line.text for block in analysis.read.blocks for line in block.lines]) if analysis.read and analysis.read.blocks else ""
                        confidence = analysis.caption.confidence if analysis.caption and analysis.caption.confidence else 0.0

                        images.append({
                            "url": image_url,
                            "description": caption,
                            "dense_captions": dense_captions,
                            "ocr_text": ocr_text,
                            "confidence": confidence
                        })
                    except Exception as e:
                        logging.error(f"Image analysis failed for attachment {attachment.get('id')}: {e}", exc_info=True)
                        continue

            return {
                "id": page_id,
                "title": page_data.get("title", "Untitled"),
                "url": page_data.get("_links", {}).get("webui", ""),
                "text": text_content,
                "images": images
            }
        except Exception as e:
            logging.error(f"Page analysis failed for page {page_id}: {e}", exc_info=True)
            return None

    def generate_architecture_report(self, software_name: str, pages: List[Dict[str, Any]], space_key: str) -> Optional[str]:
        """
        Generate a comprehensive architecture report using GPT-4.
        Returns the summary text or None if generation fails.
        """
        system_prompt = (
            f"You are a senior software architect analyzing documentation for {software_name}. "
            "Provide a detailed technical summary including:\n"
            "- System architecture and key components\n"
            "- Data flow and service interactions\n"
            "- Infrastructure and deployment patterns\n"
            "- Insights from architecture diagrams\n"
            "- Potential improvements and risks"
        )
        user_prompt = "Analyze the following documentation and images:\n\n"
        for idx, page in enumerate(pages):
            if not page:
                continue
            user_prompt += f"## Page {idx + 1}: {page.get('title')}\n"
            user_prompt += f"Content: {page.get('text')}\n"
            if page.get("images"):
                user_prompt += "Diagram Insights:\n"
                for img in page["images"]:
                    user_prompt += f"- {img.get('description')}\n"
                    dense_details = ", ".join(img.get("dense_captions", []))
                    user_prompt += f"  Detailed elements: {dense_details}\n"
                    user_prompt += f"  Extracted text: {img.get('ocr_text')}\n"
            user_prompt += "\n"

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config["azure"]["gpt_deployment"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=3000,
                top_p=0.95
            )
            summary = response.choices[0].message.content
            return summary
        except Exception as e:
            logging.error(f"GPT-4 analysis failed: {e}", exc_info=True)
            return None

    def generate_markdown_report(self, software_name: str, space_key: str, pages: List[Dict[str, Any]], summary: str) -> str:
        """
        Generate a formatted Markdown report and write it to a file.
        Returns the filename of the generated report.
        """
        report_lines = [
            f"# Software Architecture Report: {software_name}",
            f"**Confluence Space**: {space_key}",
            f"**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Analyzed Pages**: {len(pages)}",
            "",
            "## Executive Summary",
            "",
            summary,
            "",
            "## Detailed Page Analysis",
            ""
        ]

        for page in pages:
            if not page:
                continue
            report_lines.append(f"### [{page.get('title')}]({page.get('url')})")
            # Only include a snippet of the page content
            snippet = page.get("text", "")[:1000] + ("..." if len(page.get("text", "")) > 1000 else "")
            report_lines.append(f"**Key Content**:\n{snippet}\n")
            if page.get("images"):
                report_lines.append("#### Diagram Insights")
                for img in page["images"]:
                    report_lines.append(f"![Diagram]({img.get('url')})")
                    report_lines.append(f"**Description**: {img.get('description')} (Confidence: {img.get('confidence'):.0%})")
                    details = ", ".join(img.get("dense_captions", []))
                    report_lines.append(f"**Details**: {details}")
                    report_lines.append(f"**Extracted Text**: {img.get('ocr_text')}\n")

        filename = f"{software_name.replace(' ', '_')}_architecture_report.md"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            logging.info(f"Markdown report successfully written to {filename}")
        except Exception as e:
            logging.error(f"Failed to write markdown report: {e}", exc_info=True)
            raise
        return filename

    def analyze_software(self, software_name: str, space_key: str) -> Optional[str]:
        """
        End-to-end analysis workflow: search for pages, analyze each page,
        generate an architecture summary, and create a Markdown report.
        Returns the filename of the generated report or None if the process fails.
        """
        try:
            logging.info(f"Starting analysis for '{software_name}' in space '{space_key}'")
            # Step 1: Search Confluence for relevant pages
            pages_summary = self.search_confluence_space(software_name, space_key)
            if not pages_summary:
                logging.warning("No relevant pages found")
                return None

            # Step 2: Analyze pages concurrently using a thread pool
            analyzed_pages: List[Optional[Dict[str, Any]]] = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_page = {
                    executor.submit(self.analyze_page_content, page["id"]): page for page in pages_summary
                }
                for future in as_completed(future_to_page):
                    result = future.result()
                    if result:
                        analyzed_pages.append(result)

            if not analyzed_pages:
                logging.error("Failed to analyze any pages")
                return None

            # Step 3: Generate summary report using GPT-4
            summary = self.generate_architecture_report(software_name, analyzed_pages, space_key)
            if not summary:
                logging.error("Failed to generate architecture summary")
                return None

            # Step 4: Generate the Markdown report file
            report_file = self.generate_markdown_report(software_name, space_key, analyzed_pages, summary)
            logging.info(f"Report generated: {report_file}")
            return report_file

        except Exception as e:
            logging.error(f"Software analysis failed: {e}", exc_info=True)
            return None


def main() -> None:
    # Set up argument parsing for command-line usage
    parser = argparse.ArgumentParser(description="Confluence Architecture Analyzer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file")
    parser.add_argument("--software", type=str, help="Software name to analyze")
    parser.add_argument("--space", type=str, help="Confluence space key")
    args = parser.parse_args()

    try:
        config_manager = ConfigManager(args.config)
        analyzer = ConfluenceArchitectureAnalyzer(config_manager.config)

        # If software name or space key are not provided via arguments, prompt the user
        software_name = args.software or input("Enter software name to analyze: ").strip()
        space_key = args.space or input("Enter Confluence space key: ").strip()

        report_file = analyzer.analyze_software(software_name, space_key)
        if report_file:
            print(f"\nSuccess! Report saved to: {report_file}")
        else:
            print("\nAnalysis failed. Check logs for details.")
    except Exception as e:
        logging.error(f"Critical error: {e}", exc_info=True)
        print("Analysis failed. Check logs for details.")


if __name__ == "__main__":
    main()
