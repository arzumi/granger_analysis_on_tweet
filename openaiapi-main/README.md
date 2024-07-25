# Tweet Topic Classification

This project consists of three Python scripts designed to categorize a collection of tweets. The process involves generating a set of topics from the tweets, summarizing and merging these topics into a set number of final topics, and then classifying each tweet into one of these final topics.

## Prerequisites

- Python 3.x
- The following Python packages:
  - openai
  - requests
  - pandas

You can install the necessary packages using the following command:

```sh
pip install -r requirements.txt
```

# Usage

1. **Topic Extraction** - The first script takes a collection of tweets and returns a set of topics that represent the content of those tweets. This script can return many topics.

2. **Topic Summarization** - The second script uses a summarizing prompt to merge and condense the generated topics into a smaller set of final topics.

3. **Tweet Classification** - The third script takes the final summarized topics and classifies each tweet into one of these topics.

### API Keys

The scripts require access to OpenAI API. The APIs require signup and have usage costs.

1. **OpenAI API Key**: 
   - Sign up at [OpenAI](https://www.openai.com/) to get an API key.
   - Set the API key in the scripts by modifying the client initialization part:
     ```python
        from openai import OpenAI
        import pandas as pd
        import json
        from decimal import Decimal

        client = OpenAI(api_key="your-api-key")
     ```