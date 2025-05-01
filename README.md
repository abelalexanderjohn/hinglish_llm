# Hinglish - Fine-tuning LLM for Code-Switching

This project implements a workflow for fine-tuning a base language model to handle Hindi-English code-switching (Hinglish) conversations.

## Setup Instructions

### Prerequisites
- Python 3.7+
- OpenAI API key with fine-tuning access
- pip packages: `openai>=1.0.0`, `argparse`

### Environment Setup
1. Clone this repository
2. Install dependencies:
   ```
   pip install openai>=1.0.0 argparse
   ```
3. Set your OpenAI API key:
   ```
   # For Linux/Mac
   export OPENAI_API_KEY="your-api-key"
   
   # For Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key"
   ```

### Uploading Dataset
The dataset is provided as `dataset.jsonl`. No additional preparation is needed before running the fine-tuning script.

## Design Decisions

### Dataset Design (`dataset.jsonl`)

The dataset contains 20 carefully selected examples of Hinglish conversations covering different domains and code-switching patterns. This dataset was designed with the following considerations:

#### Domain Selection
- **Everyday conversations**: Greetings, weather, general chitchat
- **Task-oriented dialogues**: Setting reminders, booking services, troubleshooting
- **Information requests**: Recommendations, directions, translations
- **Customer support scenarios**: Problem-solving, assistance requests

#### Code-Switching Patterns
I've included diverse patterns of code-switching that occur naturally in Hinglish conversations:
- Hindi-dominant with English technical terms
- English-dominant with Hindi expressions for cultural concepts
- Alternating sentences in both languages
- Mid-sentence language switches
- Grammar mixing (using Hindi verbs with English nouns, etc.)

#### Stylistic Choices
- Conversational, natural-sounding responses
- Use of common Hinglish interjections and fillers
- Varying sentence lengths and complexity
- Inclusion of cultural references and phrases specific to Indian contexts
- Polite and helpful tone appropriate for an assistant

#### Size Rationale
While 20 examples is certainly a minimal dataset, it can still provide the model with enough patterns to learn basic code-switching behavior:
- Each example shows different code-switching patterns
- The examples together cover common assistant tasks
- A small dataset allows for quick iteration and testing

For production use, this dataset would need expansion to hundreds or thousands of examples across more domains.

### Model & Hyperparameter Choices

#### Model Selection
I chose `gpt-3.5-turbo` as the base model for the following reasons:
- Strong performance on conversational tasks
- More cost-effective than GPT-4 models for initial experimentation
- Better handling of multilingual content than older models (like Davinci)
- Support for chat completions format which matches our assistant use case

#### Hyperparameters
- **Epochs**: Default is 3 epochs
  - Rationale: With a small dataset, we need multiple passes to effectively learn patterns without overfitting
  - The script allows changing this via command line argument

- **Learning Rate**: Using OpenAI's default multiplier
  - Rationale: With limited data, default learning rates tend to work well
  - The script supports custom learning rate multipliers if needed for experimentation

- **Batch Size**: Using OpenAI's default
  - Rationale: OpenAI optimizes this parameter internally based on dataset size

### Prompt Formatting & Generation Settings

#### Prompt Format
We use a consistent format for all examples:
```
User: [user message in Hinglish]
Assistant:
```

This format was chosen because:
- It clearly delineates user vs. assistant roles
- It matches the chat completion format
- It provides consistent context for the model's responses

#### Generation Settings in `inference.py`
- **Temperature**: Default 0.7
  - Rationale: Provides a balance between creativity and consistency
  - Higher than 0.5 to allow for natural language variations in code-switching
  - Lower than 1.0 to maintain coherence and helpfulness

- **Max Tokens**: Default 150
  - Rationale: Allows for detailed responses while keeping them concise
  - Appropriate length for voice assistant responses (not too verbose)

Both parameters are configurable via command-line arguments for experimentation.

### Evaluation Strategy for Production

For a production Hinglish voice assistant, I recommend a multi-faceted evaluation approach:

#### Human Evaluation
- **Native Hinglish speakers** rating responses on:
  - Naturalness of code-switching
  - Appropriateness of language choice
  - Cultural relevance
  - Task completion accuracy
  
- **Target user testing** with metrics for:
  - User satisfaction
  - Task completion rate
  - Number of turns to resolution

#### Automated Metrics
- **BLEU/ROUGE** scores against human-written references (limited utility but easy to implement)
- **Perplexity** on held-out Hinglish test set
- **Code-switching consistency metrics** (e.g., measuring if Hindi/English are used in appropriate contexts)
- **Domain-specific accuracy metrics** for tasks like appointment booking, information retrieval

#### Operational Metrics
- **Fallback rates** (how often the system fails to provide a useful response)
- **Language identification accuracy** (correctly identifying Hindi vs. English segments)
- **Response latency** (important for voice applications)

#### Continuous Improvement
Implement a feedback loop where:
1. User interactions are logged (with consent)
2. A subset is sampled for human review
3. New training examples are created from successful interactions
4. The model is periodically retrained with the expanded dataset

## Usage Instructions

### Running Fine-Tuning
```bash
python fine_tune.py
```

The script will:
1. Convert the dataset to the chat format required by the OpenAI API v1.0+
2. Upload the file to OpenAI
3. Start a fine-tuning job
4. Optionally monitor the progress of the job

### Testing the Fine-Tuned Model
Once fine-tuning is complete and you have your model ID (format: `ft:gpt-3.5-turbo:custom:hinglish:abc123`), run:

```bash
python inference.py --model "your-model-id"
```

This will test the model with the hard coded test prompts.

## Sample Outputs

### Input 1
```
User: Mujhe ek chai pilao
```

### Expected Output 1
```
Zaroor! Ek garam chai lekar aati hoon. Sugar kitni daalu? Biscuit bhi chahiye?
```

### Input 2
```
User: Weather kaisa hai today in Delhi?
```

### Expected Output 2
```
Delhi ka mausam aaj pleasant hai with temperatures ranging from 28-32 degrees Celsius. Skies are mostly clear with a light breeze. Aaj ki raat ko temperatures thoda drop hokar around 25 degrees Celsius tak aa sakti hai.
```

### Input 3
```
User: Meeting reschedule karna hai, help karo
```

### Expected Output 3
```
Zarur, meeting ki details batao aur mujhe available slots batao, main phir confirm kar leti hoon.
```

## Conclusion

This fine-tuning workflow is a starting point for handling Hinglish code-switching patterns. For production use, the dataset should be significantly expanded and more rigorous evaluation procedures implemented.