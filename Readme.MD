Here’s a complete `README.md` file for your project in Markdown format. It provides all the necessary information for setting up, configuring, and using the Confluence Architecture Analyzer.

---

# Confluence Architecture Analyzer

The **Confluence Architecture Analyzer** is a Python-based tool designed to analyze software architecture documentation stored in Confluence. It uses **Azure OpenAI GPT-4** for text analysis and **Azure Computer Vision** for extracting insights from architecture diagrams and workflow images. The tool generates a detailed Markdown report summarizing the architecture, including text and image insights.

---

## Features

- **Confluence Integration**: Searches and analyzes Confluence spaces for software-specific documentation.
- **Multi-Modal Analysis**:
  - **Text Analysis**: Summarizes architecture documentation using GPT-4.
  - **Image Analysis**: Extracts insights from architecture diagrams using Azure Computer Vision (OCR, captions, and dense captions).
- **Comprehensive Reporting**: Generates a Markdown report with:
  - Executive summary
  - Detailed page analysis
  - Diagram insights with confidence scores
- **Production-Ready**:
  - Configurable via `config.yaml`
  - Comprehensive error handling and logging
  - Retry mechanisms for API calls

---

## Prerequisites

1. **Azure Resources**:
   - Azure OpenAI Service (GPT-4 deployment)
   - Azure Computer Vision
2. **Confluence Access**:
   - Confluence API token
   - Space key for the documentation
3. **Python 3.8+**
4. **Required Libraries**:
   - Install dependencies using `pip install -r requirements.txt`

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/confluence-architecture-analyzer.git
cd confluence-architecture-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `config.yaml`
Create a `config.yaml` file with the following structure:
```yaml
confluence:
  base_url: "https://your-confluence.atlassian.net/wiki/rest/api"
  user: "user@domain.com"
  api_token: "your_api_token"

azure:
  vision_endpoint: "https://your-vision-resource.cognitiveservices.azure.com"
  vision_key: "your_vision_key"
  openai_endpoint: "https://your-openai-resource.openai.azure.com"
  openai_key: "your_openai_key"
  gpt_deployment: "gpt-4"
```

### 4. Run the Tool
```bash
python confluence_analyzer.py
```

---

## Usage

### Input
- **Software Name**: The name of the software/system to analyze (e.g., "Payment Service").
- **Confluence Space Key**: The key of the Confluence space to search (e.g., "DEV").

### Output
- A Markdown report (`<software_name>_architecture_report.md`) containing:
  - Executive summary
  - Detailed page analysis
  - Diagram insights with extracted text and descriptions

---

## Example

### Input
```bash
Enter software name to analyze: Payment Service
Enter Confluence space key: DEV
```

### Output
```markdown
# Software Architecture Report: Payment Service

**Confluence Space**: DEV  
**Report Date**: 2023-10-15 14:30  
**Analyzed Pages**: 5

## Executive Summary
The Payment Service architecture is a microservices-based system with the following key components:
- **API Gateway**: Handles incoming requests and routes them to appropriate services.
- **Payment Gateway**: Integrates with external payment processors.
- **Database**: Stores transaction data in a PostgreSQL database.

### Diagram Insights
- **Architecture Diagram**: Shows a clear separation between services using Kafka for async communication.
- **Workflow Diagram**: Illustrates the payment processing flow with retry mechanisms.

## Detailed Page Analysis
### [Payment Service Overview](https://confluence/page1)
**Key Content**: The Payment Service is designed to handle high-volume transactions...
**Diagram Insights**:
- ![Diagram](https://confluence/image1.png)  
  **Description**: Architecture diagram showing service interactions (Confidence: 95%)  
  **Details**: API Gateway, Payment Service, Database  
  **Extracted Text**: "Load balancer → Docker containers"
```

---

## Configuration

### `config.yaml`
| Key | Description |
|-----|-------------|
| `confluence.base_url` | Base URL of your Confluence instance |
| `confluence.user` | Confluence user email |
| `confluence.api_token` | Confluence API token |
| `azure.vision_endpoint` | Azure Computer Vision endpoint |
| `azure.vision_key` | Azure Computer Vision key |
| `azure.openai_endpoint` | Azure OpenAI endpoint |
| `azure.openai_key` | Azure OpenAI key |
| `azure.gpt_deployment` | GPT-4 deployment name |

---

## Logs
The tool generates logs in `confluence_analyzer.log` for debugging and monitoring.

---

## Enhancements
1. **Azure Key Vault Integration**: Store credentials securely.
2. **Caching**: Cache API responses for faster repeated searches.
3. **PDF/Visio Analysis**: Add support for Azure Document Intelligence.
4. **Interactive Reports**: Generate interactive diagrams using Mermaid.js.

---

## Troubleshooting

### Common Issues
1. **API Rate Limits**:
   - Ensure your Confluence and Azure APIs have sufficient rate limits.
   - Use retry mechanisms in the code.
2. **Authentication Errors**:
   - Verify credentials in `config.yaml`.
   - Ensure the Confluence API token has read access to the space.
3. **Image Analysis Failures**:
   - Check if the image URLs are accessible.
   - Ensure Azure Computer Vision is enabled.

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

---

## Support
For questions or issues, please open an issue on GitHub or contact Vraj Routu.

---

This `README.md` provides a comprehensive guide for setting up and using the Confluence Architecture Analyzer. Let me know if you need further assistance!