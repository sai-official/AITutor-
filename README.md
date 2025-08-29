# 🎓 AI Tutor Assistant

A comprehensive AI-powered tutoring application that processes PDF documents and provides intelligent learning assistance through summarization, flashcard generation, quizzes, and interactive Q&A sessions.

## ✨ Features

### 📄 **Document Processing**
- **PDF Upload & Extraction**: Seamlessly extract text from PDF documents
- **Smart Text Processing**: Clean and normalize content using advanced NLP
- **Key Concept Extraction**: Automatically identify important terms and concepts
- **Readability Analysis**: Assess document complexity and reading level

### 🤖 **AI-Powered Learning Tools**
- **Interactive Chat**: Ask questions about your document with contextual AI responses
- **Smart Summarization**: Generate brief or detailed summaries
- **Flashcard Generation**: Create study cards from document content
- **Quiz Creation**: Generate multiple-choice questions with explanations
- **Real-time Q&A**: Get instant answers based on document understanding

### 🧠 **Advanced NLP Features**
- **Named Entity Recognition**: Identify people, places, organizations
- **TF-IDF Analysis**: Extract the most important terms and phrases
- **Similarity Matching**: Find related content and concepts
- **Definition Extraction**: Automatically detect term-definition pairs

## 🛠 Prerequisites

### System Requirements
- Python 3.8 or higher
- 8GB RAM (recommended for better performance)
- Internet connection for initial setup

### Required Software
1. **Ollama**: Local AI model server
   - Download from [https://ollama.ai](https://ollama.ai)
   - Supports Windows, macOS, and Linux

## 📦 Installation

### 1. Clone or Download
Save the `ai_tutor_app.py` file to your desired directory.

### 2. Install Python Dependencies
```bash
pip install PyPDF2 nltk ollama streamlit textstat scikit-learn
```

### 3. Install and Setup Ollama

#### Windows:
1. Download Ollama installer from [ollama.ai](https://ollama.ai)
2. Run the installer
3. Open Command Prompt and run:
```cmd
ollama pull llama3.2
```

#### macOS:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull llama3.2
```

#### Linux:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull llama3.2
```

### 4. Start Ollama Server
```bash
ollama serve
```
*Keep this terminal window open while using the app*

## 🚀 Running the Application

### 1. Start the App
```bash
streamlit run ai_tutor_app.py
```

### 2. Open Browser
The app will automatically open at `http://localhost:8501`

## 📖 How to Use

### Step 1: Upload Document
1. Click "Choose a PDF file" in the sidebar
2. Select your PDF document
3. Click "Process Document" to analyze the content

### Step 2: Explore Features

#### 💬 **Chat Tab**
- Ask questions about the document
- Get contextual answers based on content
- View chat history
- Clear conversation when needed

#### 📝 **Summary Tab**
- Choose between "brief" or "detailed" summary
- Generate AI-powered summaries
- Download summaries as text files

#### 🗂️ **Flashcards Tab**
- Set number of flashcards (5-20)
- Navigate through cards with Previous/Next
- Show/hide answers for self-testing
- Shuffle cards for varied practice
- Download all flashcards

#### 🧩 **Quiz Tab**
- Generate multiple-choice questions
- Take interactive quizzes
- Get immediate feedback with explanations
- View final scores and performance metrics

#### 📖 **Document Tab**
- View full document content
- Search within the document
- See document statistics (words, sentences, characters)

## ⚙️ Configuration

### Model Selection
Change the AI model in `ai_tutor_app.py`:
```python
def __init__(self, model_name: str = "llama3.2"):  # Change here
```

**Recommended Models:**
- `llama3.2`: General purpose, good balance
- `llama3.2:13b`: More accurate but slower
- `codellama`: Better for technical documents
- `mistral`: Alternative general-purpose model

### Performance Tuning
Adjust these parameters in the code:

#### Chunk Size (for large documents):
```python
def chunk_text(self, text: str, max_chunk_size: int = 3000):  # Increase for better context
```

#### Context Limit:
```python
self.context_limit = 4000  # Increase for longer context
```

#### Generation Parameters:
```python
options={
    'temperature': 0.7,    # 0.1-1.0 (lower = more focused)
    'top_p': 0.9,         # 0.1-1.0 (lower = more focused)
    'max_tokens': 500     # Increase for longer responses
}
```

## 🔧 Troubleshooting

### Common Issues

#### Ollama Connection Error
```
Error: Ollama connection error
```
**Solution:**
1. Ensure Ollama is installed correctly
2. Start Ollama server: `ollama serve`
3. Verify model is downloaded: `ollama list`

#### PDF Processing Error
```
Error reading PDF
```
**Solution:**
1. Ensure PDF is not password-protected
2. Try a different PDF file
3. Check if PDF contains extractable text (not just images)

#### Memory Issues
```
Out of memory error
```
**Solution:**
1. Use smaller chunk sizes
2. Reduce number of flashcards/quiz questions
3. Process smaller documents
4. Use a lighter Ollama model

#### Slow Performance
**Solutions:**
- Use smaller models (llama3.2 instead of 13b variants)
- Reduce context limits
- Process documents in smaller chunks
- Close other applications to free up RAM

### Model Installation Issues

#### Model Not Found
```bash
# List available models
ollama list

# Pull specific model
ollama pull llama3.2

# Check available models online
ollama search llama
```


## 🎯 Advanced Usage

### Custom Prompts
Modify the prompt templates for specialized content:

#### For Academic Papers:
```python
prompt = f"Analyze this academic paper and create study questions focusing on methodology, results, and implications:\n\n{chunk}"
```

#### For Technical Documentation:
```python
prompt = f"Create flashcards for this technical content, focusing on procedures, concepts, and best practices:\n\n{chunk}"
```

### Domain-Specific Fine-tuning

#### Medical Documents:
- Add medical terminology processing
- Include drug name recognition
- Focus on symptoms and treatments

#### Legal Documents:
- Extract case names and legal principles
- Focus on precedents and statutes
- Include legal definitions

#### Scientific Papers:
- Extract methodologies and results
- Focus on hypotheses and conclusions
- Include statistical information

## 📊 Performance Benchmarks

### Processing Times (Approximate)
- **10-page PDF**: 30-60 seconds
- **Summary Generation**: 15-30 seconds
- **10 Flashcards**: 45-90 seconds
- **5-question Quiz**: 30-60 seconds

### Model Comparison
| Model | Speed | Accuracy | Memory Usage |
|-------|-------|----------|--------------|
| llama3.2 | Fast | Good | 4GB |
| llama3.2:13b | Slow | Excellent | 8GB |
| mistral | Fast | Good | 4GB |
| codellama | Medium | Technical+ | 6GB |

## 🔒 Privacy & Security

- **100% Local Processing**: All data stays on your machine
- **No API Keys Required**: No external service dependencies
- **No Data Transmission**: Documents never leave your computer
- **No Storage**: Chat history cleared on restart (can be modified)

## 🛡️ Limitations

### Document Types
- **Supported**: Text-based PDFs
- **Not Supported**: Scanned PDFs (images), password-protected files
- **Best Results**: Academic papers, textbooks, articles, reports

### Performance Constraints
- Large documents (100+ pages) may require chunking
- Complex formatting may affect text extraction
- Processing time depends on hardware and model size

## 🚀 Future Enhancements

### Planned Features
- [ ] Support for DOCX and TXT files
- [ ] OCR for scanned documents
- [ ] Export to Anki format
- [ ] Progress tracking and analytics
- [ ] Multiple document comparison
- [ ] Custom study schedules

### Advanced Integrations
- [ ] Voice-based Q&A
- [ ] Image and diagram analysis
- [ ] Multi-language support
- [ ] Integration with learning management systems

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment:
```bash
python -m venv ai_tutor_env
source ai_tutor_env/bin/activate  # On Windows: ai_tutor_env\Scripts\activate
```
3. Install development dependencies
4. Make your changes
5. Test thoroughly with various PDF types

### Code Structure
- `DocumentProcessor`: Handles PDF and text processing
- `OllamaAI`: Manages all AI operations
- `AITutorApp`: Main Streamlit application
- `AdvancedNLPProcessor`: Extended NLP functionality

## 📝 Example Usage

### Basic Workflow
1. **Upload**: Select a PDF textbook chapter
2. **Process**: Let the AI analyze the content
3. **Study**: Use generated flashcards for memorization
4. **Test**: Take the quiz to assess understanding
5. **Review**: Ask specific questions in chat
6. **Export**: Download materials for offline study

### Advanced Workflow
1. **Multiple Documents**: Process related chapters/papers
2. **Concept Mapping**: Use key concepts to build connections
3. **Progressive Learning**: Start with summaries, move to details
4. **Assessment**: Regular quizzes to track progress

## 🆘 Support & FAQ

### Q: The app is slow. How can I improve performance?
A: Use lighter models (llama3.2), reduce chunk sizes, close other applications, and ensure adequate RAM.

### Q: Can I use different AI models?
A: Yes! Install any Ollama-compatible model and change the model name in the code.

### Q: How accurate are the generated questions?
A: Accuracy depends on document quality and AI model. Review generated content and adjust prompts for better results.

### Q: Can I process non-English documents?
A: The app supports any language that your chosen Ollama model can handle. Some NLTK features are English-specific.

### Q: How do I improve flashcard quality?
A: Use well-structured documents, adjust the number of cards generated, and modify the generation prompts for your specific needs.


## 🙏 Acknowledgments

- **Ollama**: For providing excellent local AI capabilities
- **NLTK**: For comprehensive natural language processing tools
- **Streamlit**: For the intuitive web application framework
- **PyPDF2**: For reliable PDF text extraction

---

**Happy Learning! 🎓📚**

For issues, suggestions, or contributions, please create an issue in the repository or reach out to the development team.